"""
eoio.readers.xml
================

Generic XML reader functionality.

This module provides a reusable :class:`XMLReader` base class for extracting
metadata from XML files using namespace-aware XPath expressions. It supports
lightweight type inference, nested text extraction, and construction of
mappings from repeated XML elements.

The class is intended to be subclassed by product- or mission-specific readers
(e.g. Sentinel-2, Landsat), which define concrete metadata paths and add
higher-level semantic accessors.
"""

import re
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Union, Callable

# Sentinel object so default=None doesn't clash with "missing container returns None"
_UNSET = object()


class XMLReader:
    """
    Generic XML metadata reader with namespace support and heuristic type casting.

    This class provides low-level utilities for reading structured values from
    XML documents. It performs namespace-aware XPath lookups using
    :attr:`metadata_paths` and applies simple heuristics to convert XML text
    content into appropriate Python scalar or sequence types.

    Subclasses are expected to define :attr:`metadata_paths` and implement
    higher-level domain-specific accessors.
    """

    #: Mapping from metadata keys to XPath expressions.
    #: Intended to be overridden by subclasses.
    metadata_paths: dict[str, str] = {}

    _INT_RE = re.compile(r"^[+-]?\d+$")
    _FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

    def __init__(self, path: Path):
        """
        Initialise the XML reader.

        :param path:
            Path to the XML file to be read.

        :raises TypeError:
            If ``path`` is ``None``.
        :raises FileNotFoundError:
            If the file does not exist.
        :raises xml.etree.ElementTree.ParseError:
            If the XML file cannot be parsed.
        """
        if path is None:
            raise TypeError("path must not be None")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self.xml_root = ET.parse(path).getroot()
        self.xml_ns = self.extract_root_namespaces(path)

    @staticmethod
    def _cast_scalar(s: str) -> Union[int, float, str]:
        """
        Cast a string token to an appropriate scalar Python type.

        Heuristic rules:
        - empty strings are returned unchanged
        - integers are returned as ``int`` (unless they have leading zeros)
        - floating-point values are returned as ``float``
        - all other values are returned as ``str``

        :param s:
            Input string token.

        :returns:
            The cast scalar value.
        """
        s = s.strip()
        if s == "":
            return ""

        if XMLReader._INT_RE.match(s):
            core = s.lstrip("+-")
            if core.startswith("0") and len(core) > 1 and core[1].isdigit():
                return s
            return int(s)

        if XMLReader._FLOAT_RE.match(s):
            core = s.lstrip("+-")
            if core.startswith("0") and len(core) > 1 and core[1].isdigit():
                return s
            return float(s)

        return s

    @staticmethod
    def _split_tokens(s: str) -> list[str]:
        """
        Split a string containing multiple values into individual tokens.

        Handles common XML encodings of numeric sequences, including:
        - whitespace- or newline-separated values
        - comma-separated values

        :param s:
            Raw string to split.

        :returns:
            List of individual value tokens.
        """
        s = s.strip()
        if not s:
            return []

        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]

        return [t for t in s.split() if t]

    @staticmethod
    def _looks_like_sequence(tokens: list[str]) -> bool:
        """
        Determine whether a list of tokens represents a sequence.

        :param tokens:
            Tokens extracted from XML text.

        :returns:
            ``True`` if more than one token is present.
        """
        return len(tokens) > 1

    @staticmethod
    def _cast_sequence(tokens: list[str]) -> list[Union[int, float, str]]:
        """
        Cast a list of string tokens to scalar Python types.

        :param tokens:
            Tokens to cast.

        :returns:
            List of cast values.
        """
        return [XMLReader._cast_scalar(t) for t in tokens]

    @staticmethod
    def _element_text_deep(elem: ET.Element) -> str:
        """
        Extract and concatenate all descendant text nodes of an XML element.

        This is useful for XML structures where values are distributed across
        multiple child elements (e.g.
        ``<Values_List><VALUES>...</VALUES></Values_List>``).

        :param elem:
            XML element from which to extract text.

        :returns:
            Concatenated text content.
        """
        parts: list[str] = []
        for t in elem.itertext():
            t = t.strip()
            if t:
                parts.append(t)
        return " ".join(parts)

    @staticmethod
    def extract_root_namespaces(xml_path: Union[str, Path]) -> dict[str, str]:
        """
        Extract namespace declarations from the root element of an XML file.

        Only namespaces declared directly on the root element are captured.

        :param xml_path:
            Path to the XML file.

        :returns:
            Mapping from namespace prefix to namespace URI.
            The default namespace (if present) is stored under ``""``.
        """
        ns_map: dict[str, str] = {}

        it = ET.iterparse(xml_path, events=("start-ns", "start"))
        for event, item in it:
            if event == "start-ns":
                prefix, uri = item  # type: ignore[misc]
                ns_map[str(prefix) or ""] = str(uri)
            elif event == "start":
                break

        return ns_map

    def find_value(
        self,
        name: str,
        *,
        default: Any = None,
        deep_text: bool = False,
        as_array: bool | None = None,
        split: str | None = "auto",
    ) -> Any:
        """
        Find and return a metadata value from the XML document.

        The value is located using the XPath associated with ``name`` in
        :attr:`metadata_paths` and converted to an appropriate Python type
        using heuristic rules.

        :param name:
            Metadata key used to look up an XPath in :attr:`metadata_paths`.
        :param default:
            Value to return if the element is missing or empty.
        :param deep_text:
            If ``True``, all descendant text nodes are used instead of only
            the element's direct text.
        :param as_array:
            Controls scalar vs sequence return:
            ``None`` = auto-detect,
            ``True`` = force list,
            ``False`` = force scalar.
        :param split:
            How to split the raw text into tokens:
              - "auto" (default): comma or whitespace (existing behaviour)
              - None: do not split; treat the entire raw text as a single token
              - any other string: split on that delimiter (e.g. "," or " ")

        :returns:
            Parsed metadata value.

        :raises KeyError:
            If ``name`` is not defined in :attr:`metadata_paths`.
        """
        if name not in self.metadata_paths:
            raise KeyError(f"Unknown metadata field: {name}")

        path = self.metadata_paths[name]
        elem = self.xml_root.find(path, self.xml_ns)

        if elem is None:
            return default

        raw = self._element_text_deep(elem) if deep_text else (elem.text or "").strip()
        if raw == "":
            return default

        if split == "auto":
            tokens = self._split_tokens(raw)
        elif split is None:
            tokens = [raw]
        else:
            tokens = [t for t in raw.split(split) if t != ""]

        if as_array is True:
            return self._cast_sequence(tokens)

        if as_array is False:
            return self._cast_scalar(tokens[0]) if tokens else default

        if self._looks_like_sequence(tokens):
            return self._cast_sequence(tokens)

        return self._cast_scalar(tokens[0])

    def find_mapping(
        self,
        name: str,
        key: str,
        *,
        key_attr: str,
        key_cast: Callable[[str], Any] = str,
        value_xpath: str | None = None,
        value_cast: Callable[[str], Any] | None = None,
        default: Any = _UNSET,
        deep_text: bool = False,
    ) -> dict[Any, Any] | None:
        """
        Build a dictionary from a parent element containing repeated child elements.

        This method supports tri-state behaviour:

        - Returns ``None`` if the parent element does not exist.
        - Returns ``{}`` if the parent exists but no child entries are found.
        - Returns a populated dictionary if child entries exist.

        Namespace robustness:
          - ``key`` is treated as a *local element name* (prefix-agnostic).
          - ``value_xpath`` (if provided) is treated as a simple slash-separated
            *local-name* path (e.g. ``"Noise_Model/ALPHA"``). Do not include
            namespace prefixes or a leading ``./``.

        :param name:
            Metadata field name mapped to a container XPath in ``self.metadata_paths``.
            The XPath should select the container element (e.g. ``.../Radiometric_Offset_List``).
        :param key:
            Local tag name of the repeated entry elements beneath the container
            (e.g. ``RADIO_ADD_OFFSET``).
        :param key_attr:
            XML attribute name to use as the dictionary key (e.g. ``bandId``).
        :param key_cast:
            Function to cast the attribute value into the desired key type.
        :param value_xpath:
            Optional local-name path (relative to each entry element) selecting the
            element whose text should be used as the dictionary value. If omitted,
            uses the entry element's own text.
        :param value_cast:
            Function to cast value text. If None, uses :meth:`_cast_scalar`.
        :param default:
            Value to return if the parent element does not exist. If not provided, the
            method returns ``None`` in that case.
        :param deep_text:
            If True, uses descendant text for the selected value element.
        :returns:
            ``None`` if the container is missing (unless ``default`` is provided),
            otherwise a dictionary (possibly empty).
        :raises KeyError:
            If ``name`` is not present in ``self.metadata_paths``.
        :raises ValueError:
            If required attributes or value elements are missing, or if values are empty.
        """
        if name not in self.metadata_paths:
            raise KeyError(f"Unknown metadata field: {name}")

        container_path = self.metadata_paths[name]

        # 1) Find container element (distinguish "missing container" vs "empty list")
        container = self.xml_root.find(container_path, self.xml_ns)
        if container is None:
            return default if default is not _UNSET else None

        def _localname(tag: str) -> str:
            # "{uri}TAG" -> "TAG", "TAG" -> "TAG"
            return tag.rsplit("}", 1)[-1]

        def _children_by_localname(parent: ET.Element, child_name: str) -> list[ET.Element]:
            return [c for c in list(parent) if _localname(c.tag) == child_name]

        def _find_first_by_localpath(parent: ET.Element, path: str) -> ET.Element | None:
            # Walk "Noise_Model/ALPHA" using local names.
            cur = parent
            for part in [p for p in path.strip("/").split("/") if p]:
                matches = _children_by_localname(cur, part)
                if not matches:
                    return None
                cur = matches[0]
            return cur

        # 2) Find repeated entry elements under the container (local-name match)
        entries = _children_by_localname(container, key)
        if not entries:
            return {}

        out: dict[Any, Any] = {}

        for elem in entries:
            if key_attr not in elem.attrib:
                raise ValueError(f"Missing attribute '{key_attr}' on element '{elem.tag}'")

            k = key_cast(elem.attrib[key_attr])

            if k in out:
                raise ValueError(f"Duplicate key {k} in mapping '{name}'")

            # Choose value element: elem itself, or a descendant found via local-name path
            value_elem = elem
            if value_xpath is not None:
                _found = _find_first_by_localpath(elem, value_xpath)
                if _found is None:
                    raise ValueError(f"Missing value element '{value_xpath}' for key {k}")
                value_elem = _found

            raw_val = self._element_text_deep(value_elem) if deep_text else (value_elem.text or "").strip()
            if raw_val == "":
                raise ValueError(f"Empty value for key {k}")

            out[k] = self._cast_scalar(raw_val) if value_cast is None else value_cast(raw_val)

        return out


if __name__ == "__main__":
    pass
