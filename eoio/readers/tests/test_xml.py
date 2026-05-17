import unittest
import tempfile
from pathlib import Path

from eoio.readers.xml import XMLReader


class _TestXMLReader(XMLReader):
    """
    Minimal concrete reader for tests: provides metadata_paths.
    """

    metadata_paths = {
        # Scalars
        "start_time": "./n1:General_Info/Product_Info/PRODUCT_START_TIME",
        "orbit_number": "./n1:General_Info/Product_Info/Datatake/SENSING_ORBIT_NUMBER",
        "processing_level": "./n1:General_Info/Product_Info/PROCESSING_LEVEL",
        "leading_zero_int": "./n1:General_Info/Product_Info/LEADING_ZERO_INT",
        "leading_zero_float": "./n1:General_Info/Product_Info/LEADING_ZERO_FLOAT",
        # Sequences
        "ext_pos_list": "./n1:Geometric_Info/Product_Footprint/Product_Footprint/Global_Footprint/EXT_POS_LIST",
        "comma_list": "./n1:General_Info/Product_Info/Comma_List",
        "spectral_values": "./n1:General_Info/Product_Info/Spectral_Response/VALUES",
        # Deep text (nested text nodes)
        "deep_values": "./n1:General_Info/Product_Info/Values_List",
        # Mapping patterns (NOTE: these now point at the *container* elements)
        "physical_gains": "./n1:General_Info/Product_Info/PHYSICAL_GAINS_LIST",
        "radio_add_offset": "./n1:General_Info/Product_Info/Radiometric_Offset_List",
        # New: mapping where value lives in a *child* element
        "parent_items": "./n1:General_Info/Product_Info/Parent_Item_List",
        # New: mapping where child value uses deep text
        "parent_deep_items": "./n1:General_Info/Product_Info/Parent_Deep_Item_List",
        # Empty container case
        "empty_list": "./n1:General_Info/Product_Info/Empty_List",
    }


class TestXMLReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Root declares n1 and xsi on the root element.
        self.xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/ns1"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="https://example.com/ns1 root.xsd">

  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>2025-11-28T11:14:31.024Z</PRODUCT_START_TIME>
      <PROCESSING_LEVEL>Level-1C</PROCESSING_LEVEL>

      <LEADING_ZERO_INT>00123</LEADING_ZERO_INT>
      <LEADING_ZERO_FLOAT>05.11</LEADING_ZERO_FLOAT>

      <Datatake datatakeIdentifier="ABC">
        <SENSING_ORBIT_NUMBER>137</SENSING_ORBIT_NUMBER>
      </Datatake>

      <Comma_List>1, 2, 3</Comma_List>

      <Spectral_Response>
        <VALUES>0.1 0.2 0.3</VALUES>
      </Spectral_Response>

      <Values_List>
        <VALUES>1 2</VALUES>
        <VALUES>3</VALUES>
      </Values_List>

      <!-- MAPPINGS: now wrapped in container elements -->
      <PHYSICAL_GAINS_LIST>
        <PHYSICAL_GAINS bandId="0">4.0867035</PHYSICAL_GAINS>
        <PHYSICAL_GAINS bandId="1">3.74665902</PHYSICAL_GAINS>
      </PHYSICAL_GAINS_LIST>

      <Radiometric_Offset_List>
        <RADIO_ADD_OFFSET band_id="0">-1000</RADIO_ADD_OFFSET>
        <RADIO_ADD_OFFSET band_id="1">-1000</RADIO_ADD_OFFSET>
      </Radiometric_Offset_List>

      <!-- New: value in child element (for value_xpath tests) -->
      <Parent_Item_List>
        <Parent_Item bandId="0">
          <VAL>10</VAL>
        </Parent_Item>
        <Parent_Item bandId="1">
          <VAL>20</VAL>
        </Parent_Item>
      </Parent_Item_List>

      <!-- New: value in child element but split across multiple descendants -->
      <Parent_Deep_Item_List>
        <Parent_Deep_Item bandId="0">
          <VALS>
            <V>1</V>
            <V>2</V>
          </VALS>
        </Parent_Deep_Item>
        <Parent_Deep_Item bandId="1">
          <VALS>
            <V>3</V>
          </VALS>
        </Parent_Deep_Item>
      </Parent_Deep_Item_List>

      <!-- Empty container (container exists, no entries) -->
      <Empty_List></Empty_List>

    </Product_Info>
  </n1:General_Info>

  <n1:Geometric_Info>
    <Product_Footprint>
      <Product_Footprint>
        <Global_Footprint>
          <EXT_POS_LIST>54.0 -1.0 53.0 0.0 54.0 -1.0</EXT_POS_LIST>
        </Global_Footprint>
      </Product_Footprint>
    </Product_Footprint>
  </n1:Geometric_Info>

</n1:Root>
"""
        self.xml_file = self.tmp_path / "test.xml"
        self.xml_file.write_text(self.xml_text, encoding="utf-8")
        self.reader = _TestXMLReader(self.xml_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    # ---- __init__ ------------------------------------------------------------

    def test_init_raises_for_none_path(self):
        with self.assertRaises(TypeError):
            _TestXMLReader(None)  # type: ignore[arg-type]

    def test_init_raises_for_missing_file(self):
        missing = self.tmp_path / "missing.xml"
        with self.assertRaises(FileNotFoundError):
            _TestXMLReader(missing)

    # ---- Namespace handling --------------------------------------------------

    def test_extract_root_namespaces(self):
        ns = self.reader.xml_ns
        self.assertIn("n1", ns)
        self.assertEqual(ns["n1"], "https://example.com/ns1")

        self.assertIn("xsi", ns)
        self.assertEqual(ns["xsi"], "http://www.w3.org/2001/XMLSchema-instance")

    # ---- find_value: scalar coercion ----------------------------------------

    def test_find_value_string(self):
        v = self.reader.find_value("processing_level", split=None)  # ensure not split
        self.assertIsInstance(v, str)
        self.assertEqual(v, "Level-1C")

    def test_find_value_int(self):
        v = self.reader.find_value("orbit_number")
        self.assertIsInstance(v, int)
        self.assertEqual(v, 137)

    def test_find_value_datetime_like_remains_string(self):
        v = self.reader.find_value("start_time", split=None)
        self.assertIsInstance(v, str)
        self.assertEqual(v, "2025-11-28T11:14:31.024Z")

    def test_find_value_leading_zero_int_remains_string(self):
        v = self.reader.find_value("leading_zero_int", split=None)
        self.assertIsInstance(v, str)
        self.assertEqual(v, "00123")

    def test_find_value_leading_zero_float_remains_string(self):
        v = self.reader.find_value("leading_zero_float", split=None)
        self.assertIsInstance(v, str)
        self.assertEqual(v, "05.11")

    # ---- find_value: sequence coercion --------------------------------------

    def test_find_value_whitespace_separated_list(self):
        v = self.reader.find_value("spectral_values")
        self.assertIsInstance(v, list)
        self.assertEqual(v, [0.1, 0.2, 0.3])

    def test_find_value_comma_separated_list(self):
        v = self.reader.find_value("comma_list")
        self.assertIsInstance(v, list)
        self.assertEqual(v, [1, 2, 3])

    def test_find_value_ext_pos_list(self):
        v = self.reader.find_value("ext_pos_list")
        self.assertIsInstance(v, list)
        self.assertEqual(v, [54.0, -1.0, 53.0, 0.0, 54.0, -1.0])

    # ---- deep_text behaviour -------------------------------------------------

    def test_find_value_deep_text_joins_descendants(self):
        v1 = self.reader.find_value("deep_values", default=None, deep_text=False)
        self.assertIsNone(v1)

        v2 = self.reader.find_value("deep_values", default=None, deep_text=True)
        self.assertEqual(v2, [1, 2, 3])

    # ---- as_array overrides --------------------------------------------------

    def test_as_array_true_forces_list(self):
        v = self.reader.find_value("orbit_number", as_array=True)
        self.assertIsInstance(v, list)
        self.assertEqual(v, [137])

    def test_as_array_false_forces_scalar(self):
        v = self.reader.find_value("spectral_values", as_array=False)
        self.assertIsInstance(v, float)
        self.assertEqual(v, 0.1)

    # ---- error paths & defaults ---------------------------------------------

    def test_unknown_metadata_name_raises(self):
        with self.assertRaises(KeyError):
            self.reader.find_value("does_not_exist")

    def test_missing_element_returns_default(self):
        class _ReaderMissing(XMLReader):
            metadata_paths = {"missing": "./n1:General_Info/Product_Info/NOPE"}

        r = _ReaderMissing(self.xml_file)
        self.assertEqual(r.find_value("missing", default="fallback"), "fallback")

    # ---- find_mapping: updated behaviour ------------------------------------

    def test_find_mapping_casts_keys_and_values(self):
        gains = self.reader.find_mapping(
            "physical_gains",
            key="PHYSICAL_GAINS",
            key_attr="bandId",
            key_cast=int,
            value_cast=float,
        )
        self.assertEqual(gains, {0: 4.0867035, 1: 3.74665902})

    def test_find_mapping_uses_cast_scalar_when_value_cast_none(self):
        offsets = self.reader.find_mapping(
            "radio_add_offset",
            key="RADIO_ADD_OFFSET",
            key_attr="band_id",
            key_cast=int,
            value_cast=None,  # should use _cast_scalar -> ints
        )
        self.assertEqual(offsets, {0: -1000, 1: -1000})
        self.assertIsInstance(offsets[0], int)

    def test_find_mapping_returns_default_when_container_missing(self):
        class _ReaderMissing(XMLReader):
            metadata_paths = {"missing_container": "./n1:General_Info/Product_Info/DOES_NOT_EXIST"}

        r = _ReaderMissing(self.xml_file)
        out = r.find_mapping("missing_container", key="ITEM", key_attr="id", default={"x": "y"})
        self.assertEqual(out, {"x": "y"})

    def test_find_mapping_returns_none_when_container_missing_and_no_default(self):
        class _ReaderMissing(XMLReader):
            metadata_paths = {"missing_container": "./n1:General_Info/Product_Info/DOES_NOT_EXIST"}

        r = _ReaderMissing(self.xml_file)
        out = r.find_mapping("missing_container", key="ITEM", key_attr="id")
        self.assertIsNone(out)

    def test_find_mapping_returns_empty_dict_when_container_exists_but_no_entries(self):
        out = self.reader.find_mapping(
            "empty_list",
            key="EMPTY_ITEM",
            key_attr="id",
        )
        self.assertEqual(out, {})

    # ---- find_mapping: value_xpath behaviour --------------------------------

    def test_find_mapping_value_xpath_reads_child_text(self):
        out = self.reader.find_mapping(
            "parent_items",
            key="Parent_Item",
            key_attr="bandId",
            key_cast=int,
            value_xpath="VAL",  # local-name path (no ./, no prefixes)
            value_cast=int,
        )
        self.assertEqual(out, {0: 10, 1: 20})

    def test_find_mapping_value_xpath_missing_child_raises(self):
        # IMPORTANT: container must exist and contain entries for this test,
        # otherwise find_mapping will return None or {} before it ever checks value_xpath.
        bad_xml = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/ns1">
  <n1:General_Info>
    <Product_Info>
      <Parent_Item_List>
        <Parent_Item bandId="0"></Parent_Item>
      </Parent_Item_List>
    </Product_Info>
  </n1:General_Info>
</n1:Root>
"""
        xml_file = self.tmp_path / "bad_value_xpath.xml"
        xml_file.write_text(bad_xml, encoding="utf-8")
        r = _TestXMLReader(xml_file)

        with self.assertRaises(ValueError):
            r.find_mapping(
                "parent_items",
                key="Parent_Item",
                key_attr="bandId",
                key_cast=int,
                value_xpath="VAL",
                value_cast=int,
            )

    def test_find_mapping_value_xpath_deep_text(self):
        # VALS contains multiple <V> children; deep_text should join them.
        def _vals_cast(s: str) -> list[int]:
            return [int(t) for t in s.split()]

        out = self.reader.find_mapping(
            "parent_deep_items",
            key="Parent_Deep_Item",
            key_attr="bandId",
            key_cast=int,
            value_xpath="VALS",
            value_cast=_vals_cast,
            deep_text=True,
        )
        self.assertEqual(out, {0: [1, 2], 1: [3]})

    # ---- find_mapping: error paths ------------------------------------------

    def test_find_mapping_raises_on_missing_key_attr(self):
        # Must include the container + an entry matching `key`, but omit key_attr.
        bad_xml = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/ns1">
  <n1:General_Info>
    <Product_Info>
      <PHYSICAL_GAINS_LIST>
        <PHYSICAL_GAINS>4.0</PHYSICAL_GAINS>
      </PHYSICAL_GAINS_LIST>
    </Product_Info>
  </n1:General_Info>
</n1:Root>
"""
        xml_file = self.tmp_path / "bad.xml"
        xml_file.write_text(bad_xml, encoding="utf-8")
        r = _TestXMLReader(xml_file)

        with self.assertRaises(ValueError):
            r.find_mapping(
                "physical_gains",
                key="PHYSICAL_GAINS",
                key_attr="bandId",
                key_cast=int,
                value_cast=float,
            )

    def test_find_mapping_raises_on_empty_value(self):
        # Must include the container + an entry with key_attr, but empty value.
        bad_xml = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Root xmlns:n1="https://example.com/ns1">
  <n1:General_Info>
    <Product_Info>
      <PHYSICAL_GAINS_LIST>
        <PHYSICAL_GAINS bandId="0"></PHYSICAL_GAINS>
      </PHYSICAL_GAINS_LIST>
    </Product_Info>
  </n1:General_Info>
</n1:Root>
"""
        xml_file = self.tmp_path / "bad2.xml"
        xml_file.write_text(bad_xml, encoding="utf-8")
        r = _TestXMLReader(xml_file)

        with self.assertRaises(ValueError):
            r.find_mapping(
                "physical_gains",
                key="PHYSICAL_GAINS",
                key_attr="bandId",
                key_cast=int,
                value_cast=float,
            )


if __name__ == "__main__":
    unittest.main()
