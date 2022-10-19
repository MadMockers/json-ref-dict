from collections import abc
from functools import lru_cache
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union, Set
from urllib.parse import unquote

from jsonpointer import JsonPointer, JsonPointerException, _nothing

from json_ref_dict.exceptions import DocumentParseError
from json_ref_dict.loader import get_document, JSONSchema
from json_ref_dict.uri import URI, parse_segment

ResolvedValue = Union[List, Dict, None, bool, str, int, float]
UriValuePair = Tuple[URI, ResolvedValue]


class RefPointer(JsonPointer):
    """Subclass of `JsonPointer` which accepts full URI.

    When references are encountered, resolution is deferred to a new
    `RefPointer` with a new URI context to support reference nesting.
    """

    def __init__(self, uri: Union[str, URI]):
        self.uri = URI.from_string(uri) if isinstance(uri, str) else uri
        super().__init__(self.uri.pointer)


    def resolve(self, doc: Any, default: Any = _nothing) -> Any:
        """Resolves the pointer against doc and returns the referenced object.

        If any remotes are found, then resolution is deferred to a new
        pointer instance in that reference scope and resolving continued,
        until the value is found.

        :param doc: The document in which to start resolving the pointer.
        :param default: The value to return if the pointer fails. If not
            passed, an exception will be raised.
        :return: The value of the pointer in the containing document.
            The document may be different from the one given in as an argument.
        :raises JsonPointerException: if `default` is not set and pointer
            could not be resolved.
        """
        _, value = _resolve_itr(self.uri, default=default, doc=doc)
        return value

    get = resolve

    def resolve_with_uri(
        self, doc: Any, default: Any = _nothing
    ) -> Tuple[URI, Any]:
        """Resolves the pointer, starting from given doc

        The resolver recurses references and, as a result, may end up on
        a different document than the one given in as an argument. The URI for
        the document containing the value is returned together with the value.

        :param doc: The document with which to start resolving the pointer.
        :param default: The value to return if the pointer fails. If not
            passed, an exception will be raised.
        :return: The updated URI and the value of the pointer
            in the located document.
        :raises JsonPointerException: if `default` is not set and pointer
            could not be resolved.
        """
        return _resolve_itr(self.uri, default=default, doc=doc)

    def set(self, doc: Any, value: Any, inplace: bool = True) -> NoReturn:
        """`RefPointer` is read-only."""
        raise NotImplementedError("Cannot edit distributed schema.")

    def to_last(self, doc: Any) -> Tuple[Any, Union[str, int]]:
        """Resolves pointer until the last step.

        :return: (sub-doc, last-step).
        """
        result = RefPointer(self.uri.back()).resolve(doc)
        part = self.get_part(result, self.parts[-1])
        return result, part


def nested_lru_cache(nested_cache_funcs, *args, **kwargs):
    """Implement lru_cache that also resets the underlying document cache.

    URI resolver functions share a common document cache that
    is implemented with lru_cache.
    Cache clearing on the resolve-functions needs to reset
    the underlying cache with a custom cache_clear implementation.
    """

    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        orig_cache_clear = func.cache_clear

        def _custom_clear():
            for func in nested_cache_funcs:
                func.cache_clear()
            orig_cache_clear()

        func.cache_clear = _custom_clear
        return func

    return decorator


@lru_cache(maxsize=None)
def _resolve_cached_root_doc(uri: URI) -> JSONSchema:
    try:
        return get_document(uri.root)
    except DocumentParseError as exc:
        raise DocumentParseError(
            f"Failed to load base document of {uri}."
        ) from exc


@nested_lru_cache([_resolve_cached_root_doc], maxsize=None)
def resolve_uri_to_urivalue_pair(uri: URI) -> UriValuePair:
    """Find the value-containing URI and actual value for a given starting URI.

    The starting URI can point to another document than where the final
    value is found. This method returns a valid URI to the containing document.

    Loads the document and resolves the pointer, bypassing refs.
    Utilises `lru_cache` to avoid re-loading multiple documents.

    :return: URI to the document containing the value, and the value itself.

    :raises DocumentParseError: if the input URI root does not point to
        a valid document.
    """
    return _resolve_itr(uri)


@nested_lru_cache([resolve_uri_to_urivalue_pair], maxsize=None)
def resolve_uri(uri: URI) -> ResolvedValue:
    """Find the value for a given URI.

    Loads the document and resolves the pointer, bypassing refs.
    """
    #_, value = resolve_uri_to_urivalue_pair(uri)
    #return value
    _, value = _resolve_itr(uri)
    return value


def _walker(doc: Any, uri: URI, parts: List[str]):

    parts = [p for p in parts if p]

    yield uri, parts, doc

    for idx, part in enumerate(parts):
        try:
            uri = uri.get(parse_segment(part))
            ref = RefPointer(uri)
            try:
                doc = ref.walk(doc, part)
            except JsonPointerException:
                doc = ref.walk(doc, unquote(part))

            yield uri, parts[idx+1:], doc
        except JsonPointerException as e:
            yield uri, parts[idx+1:], e


def _resolve_links(links: Dict[URI, URI], uri: URI, visited: Set[URI] = None):
    if visited is None:
        visited = set()

    if uri in visited:
        raise RuntimeError("Self reference")

    if uri in links:
        visited.add(uri)
        return True, _resolve_links(links, links[uri], visited=visited)[1]
    else:
        return False, uri


def _resolve_itr(uri: URI, doc: Any = None, default: Any = _nothing):
    visited: Dict[URI, Any] = {}
    links: Dict[URI, URI] = {}

    if doc is None:
        doc = _resolve_cached_root_doc(uri)

    ref   = RefPointer(uri)
    parts = ref.parts

    uri   = uri.get("/")

    skip = 0
    while True:
        for uri, remaining, doc in _walker(doc, uri, parts):
            if isinstance(doc, JsonPointerException):
                if default is _nothing:
                    raise doc
                else:
                    return uri, default

            if skip > 0:
                skip -= 1
                continue

            prev = uri
            uri_changed, uri = _resolve_links(links, uri)
            if uri_changed:
                doc   = visited[uri]
                parts = remaining
                break

            if uri not in visited:
                visited[uri] = doc

            if isinstance(doc, abc.Mapping):
                ref = doc.get("$ref")
                if isinstance(ref, str):
                    ref_uri    = uri.relative(ref)

                    if uri.contains(ref_uri):
                        # The reference we've come across is actually inside
                        # the document we're currently looking at
                        refp  = RefPointer(ref_uri)
                        parts = refp.parts + remaining
                        skip  = 1
                        break

                    else:
                        links[uri] = ref_uri

                        if ref_uri in visited:
                            doc   = visited[ref_uri]
                            uri   = ref_uri
                            parts = remaining
                        else:
                            doc   = _resolve_cached_root_doc(ref_uri)
                            uri   = ref_uri.get("/")
                            refp  = RefPointer(ref_uri)
                            parts = refp.parts + remaining
                    break

            if len(remaining) == 0:
                return uri, doc
