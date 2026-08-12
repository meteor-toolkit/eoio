"""eoio.readers.sentinel2.layout - helper for Sentinel-2 SAFE layout parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os
import re


class S2LayoutError(ValueError):
    """Raised when a SAFE layout is missing expected files/structure."""


_BAND_TOKEN_RE = re.compile(r"B(?P<band>\d{1,2}A?)")
_L2A_LAYER_TOKEN_RE = re.compile(r"(?P<layer>AOT|WVP|SCL|TCI)")
_QI_TOKEN_RE = re.compile(r"(?P<qi>CLDPRB|SNWPRB)")

_IMG_DATA_DIR_RE = re.compile(r"IMG_DATA(?:_R(?P<res>\d{2})m)?$")
_JP2_RES_RE = re.compile(r"_(?P<res>\d{2})m\.jp2$", flags=re.IGNORECASE)


DEFAULT_IMG_RES_M: Dict[str, int] = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
    "AOT": 10,
    "WVP": 10,
    "TCI": 10,
    "SCL": 20,
}


@dataclass(frozen=True)
class S2Layout:
    """
    SAFE layout helper for Sentinel-2 MSI products (L1C/L2A).

    Solely responsible for filesystem/path logic. No heavy dependencies.
    """

    safe_path: str

    def __post_init__(self) -> None:
        """
        Validate SAFE path.
        """
        p = Path(self.safe_path)
        if not p.exists():
            raise S2LayoutError(f"SAFE path not found: {self.safe_path}")
        if not p.is_dir():
            raise S2LayoutError(f"SAFE path is not a directory: {self.safe_path}")

    @property
    def proc_version(self) -> Tuple[int, int]:
        """
        Return processing version number as a tuple.

        :return:
            Tuple of major version and minor version.
        """
        return (
            int(os.path.split(self.safe_path)[-1].split("_")[3][1:][:2]),
            int(os.path.split(self.safe_path)[-1].split("_")[3][1:][-2:]),
        )

    @property
    def safe_dir(self) -> Path:
        """
        Return SAFE path as a pathlib Path.

        :return:
            SAFE directory path.
        """
        return Path(self.safe_path)

    @property
    def processing_level_guess(self) -> str:
        """
        Best-effort processing level inference from SAFE name / metadata presence.

        :return:
            ``"L2A"`` when the SAFE appears to be Level-2A, otherwise ``"L1C"``.
        """
        name = self.safe_dir.name.upper()
        try:
            mtd = self.product_metadata_xml().name.upper()  # type: ignore[union-attr]
        except Exception:
            mtd = ""
        if "MSIL2A" in name or mtd == "MTD_MSIL2A.XML":
            return "L2A"
        return "L1C"

    def qi_data_dir(self, granule: Optional[Path] = None) -> Optional[Path]:
        """
        Return QI_DATA directory if present under the granule.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            Path to QI_DATA directory, or ``None`` if not found.
        """
        g = granule or self.granule_dir()
        cand = g / "QI_DATA"
        if cand.exists():
            return cand
        deep = sorted(g.rglob("QI_DATA"))
        return deep[0] if deep else None

    def qi_jp2_files(self, granule: Optional[Path] = None) -> List[Path]:
        """
        Return JP2 files found under QI_DATA.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            List of QI_DATA JP2 files.
        """
        qidir = self.qi_data_dir(granule=granule)
        if qidir is None:
            return []

        files = list(sorted(qidir.glob("*.jp2"))) + list(sorted(qidir.rglob("*.jp2")))
        seen = set()
        uniq: List[Path] = []
        for f in files:
            if f.is_file() and f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq

    def msk_classi_path(self, granule: Optional[Path] = None) -> Optional[str]:
        """
        Return the path to the L1C classification mask (``MSK_CLASSI_B00.jp2``), if present.

        This is the per-granule cloud/snow classification raster shipped with
        L1C products from processing baseline 04.00 onwards: a 60 m, 3-band
        JP2 whose bands are opaque clouds, cirrus and snow/ice (see
        :py:data:`~eoio.readers.sentinel2.masks.MSK_CLASSI_BANDS`). Earlier
        baselines instead provide ``QA60.jp2`` / ``MSK_CLOUDS_B00.gml``, which
        are not handled here -- callers get ``None`` and should treat masks as
        unavailable for those products.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            Path to the classification mask JP2, or ``None`` if the product
            does not carry one.
        """
        for f in self.qi_jp2_files(granule=granule):
            if "MSK_CLASSI" in f.name.upper():
                return str(f)
        return None

    def available_img_tokens(self) -> List[str]:
        """
        Return unique IMG_DATA layer tokens inferred from filenames.

        Includes band tokens and L2A layer tokens.

        :return:
            Sorted list of available IMG_DATA tokens.
        """
        tokens = set()
        for f in self.jp2_files():
            token = self._token_from_jp2_name(f.name)
            if token is not None:
                tokens.add(token)

        def _sort(token: str) -> Tuple[int, int, int, str]:
            if token.startswith("B"):
                band = token[1:]
                if band.endswith("A"):
                    return (0, int(band[:-1]), 1, token)
                return (0, int(band), 0, token)
            return (1, 0, 0, token)

        return sorted(tokens, key=_sort)

    def available_qi_tokens(self) -> List[str]:
        """
        Return unique QI_DATA token names.

        :return:
            Sorted list of QI_DATA token names.
        """
        toks = set()
        for f in self.qi_jp2_files():
            m = _QI_TOKEN_RE.search(f.name.upper())
            if m:
                toks.add(m.group("qi"))
        return sorted(toks)

    def qi_jp2_paths(self, qi_vars: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """
        Return mapping from QI variable name to JP2 filepath.

        :param qi_vars:
            Optional list of requested QI variable names. If omitted, all
            available QI variables are returned.
        :return:
            Mapping ``{qi_var: filepath}``.
        :raises S2LayoutError:
            If any requested variable is not present.
        """
        if qi_vars is None:
            qi_vars = self.available_qi_tokens()

        index: Dict[str, List[Path]] = {}
        for f in self.qi_jp2_files():
            m = _QI_TOKEN_RE.search(f.name.upper())
            if not m:
                continue
            tok = m.group("qi")
            index.setdefault(tok, []).append(f)

        out: Dict[str, str] = {}
        missing: List[str] = []

        for v in qi_vars:
            vnorm = v.upper()
            cands = index.get(vnorm, [])
            if not cands:
                missing.append(v)
                continue
            out[v] = str(sorted(cands, key=lambda p: (len(p.parts), str(p)))[0])

        if missing:
            raise S2LayoutError(
                "Requested QI vars not found in SAFE: "
                + ", ".join(missing)
                + f". Available: {', '.join(self.available_qi_tokens())}"
            )

        return out

    def granule_dirs(self) -> List[Path]:
        """
        Return GRANULE subdirectories.

        :return:
            List of granule directories.
        :raises S2LayoutError:
            If GRANULE directory or granule subdirectories are missing.
        """
        gdir = self.safe_dir / "GRANULE"
        if not gdir.exists():
            raise S2LayoutError(f"Expected GRANULE directory not found under: {self.safe_dir}")

        granules = sorted([p for p in gdir.iterdir() if p.is_dir()])
        if not granules:
            raise S2LayoutError(f"No granule directories found under: {gdir}")
        return granules

    def granule_dir(self) -> Path:
        """
        Return the primary granule directory.

        :return:
            Primary granule directory.
        """
        return self.granule_dirs()[0]

    def datastrip_dir(self, datastrip_id: Optional[str] = None) -> Path:
        """
        Return the Sentinel-2 datastrip directory.

        :param datastrip_id:
            Optional datastrip identifier. If omitted, the SAFE must contain
            exactly one datastrip.
        :return:
            Datastrip directory path.
        :raises S2LayoutError:
            If the datastrip directory cannot be resolved.
        """
        base = Path(self.safe_path) / "DATASTRIP"
        if not base.is_dir():
            raise S2LayoutError(f"SAFE is missing DATASTRIP directory: {base}")

        if datastrip_id:
            p = base / datastrip_id
            if not p.is_dir():
                raise S2LayoutError(f"Datastrip directory not found for datastrip '{datastrip_id}': {p}")
            return p

        candidates = sorted(p for p in base.iterdir() if p.is_dir())
        if not candidates:
            raise S2LayoutError(f"No datastrip directories found under: {base}")

        if len(candidates) > 1:
            raise S2LayoutError(
                "Multiple datastrip directories found. Specify datastrip_id. "
                f"Candidates: {', '.join(p.name for p in candidates)}"
            )

        return candidates[0]

    def product_metadata_xml(self) -> Optional[Path]:
        """
        Return product-level metadata XML.

        :return:
            Product metadata XML path, or ``None`` if not found.
        """
        cands = sorted(self.safe_dir.glob("MTD_MSI*.xml"))
        return cands[0] if cands else None

    def tl_metadata_xml(self, granule: Optional[Path] = None) -> Path:
        """
        Return tile-level metadata XML.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            Tile metadata XML path.
        :raises S2LayoutError:
            If tile metadata cannot be found.
        """
        g = granule or self.granule_dir()
        cands = sorted(g.glob("MTD_TL*.xml")) + sorted(g.glob("MTD_TL.xml"))
        for c in cands:
            if c.exists():
                return c

        deep = sorted(g.rglob("MTD_TL*.xml"))
        if deep:
            return deep[0]

        raise S2LayoutError(f"Tile-level metadata XML not found under granule: {g}")

    def ds_metadata_xml(self, datastrip_id: Optional[str] = None) -> Path:
        """
        Return the Sentinel-2 datastrip metadata file.

        :param datastrip_id:
            Optional datastrip identifier.
        :return:
            Path to ``MTD_DS.xml``.
        :raises S2LayoutError:
            If the datastrip metadata file is missing.
        """
        ds_dir = self.datastrip_dir(datastrip_id=datastrip_id)

        p = ds_dir / "MTD_DS.xml"
        if not p.is_file():
            raise S2LayoutError(f"MTD_DS.xml not found in datastrip directory: {ds_dir}")

        return p

    def img_data_dirs(
        self,
        granule: Optional[Path] = None,
    ) -> List[Tuple[Optional[int], Path]]:
        """
        Find IMG_DATA folders and return ``(resolution_m, path)`` pairs.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            List of ``(resolution_m, path)`` pairs.
        :raises S2LayoutError:
            If no IMG_DATA directories are found.
        """
        g = granule or self.granule_dir()

        candidates: List[Path] = []
        for base in [g, g / "IMG_DATA", g / "IMG_DATA" / "R10m"]:
            if base.exists():
                candidates.append(base)

        candidates.extend(sorted(g.rglob("IMG_DATA")))
        candidates.extend(sorted(g.rglob("IMG_DATA_R10m")))
        candidates.extend(sorted(g.rglob("IMG_DATA_R20m")))
        candidates.extend(sorted(g.rglob("IMG_DATA_R60m")))

        seen = set()
        uniq: List[Path] = []
        for p in candidates:
            if p.is_dir() and p not in seen:
                seen.add(p)
                uniq.append(p)

        out: List[Tuple[Optional[int], Path]] = []
        for p in uniq:
            m = _IMG_DATA_DIR_RE.search(p.name)
            res: Optional[int] = None
            if m and m.group("res"):
                res = int(m.group("res"))
            out.append((res, p))

        for p in sorted((g / "IMG_DATA").glob("R*m")) if (g / "IMG_DATA").exists() else []:
            if p.is_dir():
                try:
                    res = int(p.name.replace("R", "").replace("m", ""))
                except ValueError:
                    res = None
                out.append((res, p))

        final: List[Tuple[Optional[int], Path]] = []
        seen2 = set()
        for res, p in out:
            key = (res, str(p))
            if key not in seen2:
                seen2.add(key)
                final.append((res, p))

        final.sort(key=lambda t: (t[0] is None, t[0] or 0, str(t[1])))
        if not final:
            raise S2LayoutError(f"No IMG_DATA directories found under granule: {g}")

        return final

    def jp2_files(self, granule: Optional[Path] = None) -> List[Path]:
        """
        Return all JP2 files found under IMG_DATA directories.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            List of IMG_DATA JP2 files.
        :raises S2LayoutError:
            If no JP2 files are found.
        """
        files: List[Path] = []
        for _, d in self.img_data_dirs(granule=granule):
            files.extend(sorted(d.glob("*.jp2")))
            files.extend(sorted(d.rglob("*.jp2")))

        seen = set()
        uniq: List[Path] = []
        for f in files:
            if f.is_file() and f not in seen:
                seen.add(f)
                uniq.append(f)

        if not uniq:
            raise S2LayoutError(f"No JP2 imagery found under IMG_DATA for: {self.safe_dir}")

        return uniq

    def available_band_tokens(self) -> List[str]:
        """
        Return unique band tokens inferred from IMG_DATA filenames.

        :return:
            List of available band tokens.
        """
        return [t for t in self.available_img_tokens() if t.startswith("B")]

    def default_meas_vars(self) -> List[str]:
        """
        Return default measurement variables to read.

        :return:
            Default measurement variable list.
        """
        bands = self.available_band_tokens()
        if self.processing_level_guess == "L2A":
            extras = [t for t in ("AOT", "WVP", "SCL") if t in self.available_img_tokens()]
            return bands + extras
        return bands

    @staticmethod
    def _token_from_jp2_name(name: str) -> Optional[str]:
        """
        Extract IMG_DATA token from a JP2 filename.

        :param name:
            JP2 filename.
        :return:
            Token such as ``"B02"``, ``"AOT"``, or ``None`` if not recognised.
        """
        name_u = name.upper()

        m = _BAND_TOKEN_RE.search(name_u)
        if m:
            token = f"B{m.group('band')}"
            if token != "B00":
                return token

        m = _L2A_LAYER_TOKEN_RE.search(name_u)
        if m:
            return m.group("layer")

        return None

    @staticmethod
    def _resolution_from_path(path: Path) -> Optional[int]:
        """
        Extract raster resolution in metres from a JP2 path.

        :param path:
            JP2 path.
        :return:
            Resolution in metres, or ``None`` if it cannot be determined.
        """
        m = _JP2_RES_RE.search(path.name)
        if m:
            return int(m.group("res"))

        parent_name = path.parent.name.upper()
        if parent_name.startswith("R") and parent_name.endswith("M"):
            try:
                return int(parent_name[1:-1])
            except ValueError:
                pass

        if "IMG_DATA_R10M" in path.parts:
            return 10
        if "IMG_DATA_R20M" in path.parts:
            return 20
        if "IMG_DATA_R60M" in path.parts:
            return 60

        return None

    def _select_img_path_for_token(
        self,
        token: str,
        candidates: List[Path],
        prefer_res_m: Optional[int] = None,
    ) -> Path:
        """
        Select the most appropriate IMG_DATA JP2 path for a token.

        Selection order is:
        1. preferred resolution, if requested and available
        2. default/native resolution for the token, if available
        3. closest available resolution to native/default
        4. stable lexical fallback

        :param token:
            IMG_DATA token such as ``"B02"`` or ``"AOT"``.
        :param candidates:
            Candidate JP2 paths for the token.
        :param prefer_res_m:
            Optional preferred resolution in metres.
        :return:
            Selected JP2 path.
        :raises S2LayoutError:
            If no candidates are available.
        """
        if not candidates:
            raise S2LayoutError(f"No IMG_DATA JP2 candidates found for token: {token}")

        by_res: Dict[Optional[int], List[Path]] = {}
        for path in candidates:
            res = self._resolution_from_path(path)
            by_res.setdefault(res, []).append(path)

        if prefer_res_m is not None and prefer_res_m in by_res:
            return sorted(by_res[prefer_res_m])[0]

        native_res = DEFAULT_IMG_RES_M.get(token.upper())
        if native_res is not None and native_res in by_res:
            return sorted(by_res[native_res])[0]

        known_res = [r for r in by_res.keys() if r is not None]
        if native_res is not None and known_res:
            best_res = min(known_res, key=lambda r: abs(r - native_res))
            return sorted(by_res[best_res])[0]

        return sorted(
            candidates,
            key=lambda p: (
                self._resolution_from_path(p) is None,
                self._resolution_from_path(p) or 9999,
                str(p),
            ),
        )[0]

    def img_jp2_paths(
        self,
        meas_vars: Optional[Sequence[str]] = None,
        prefer_res_m: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Return mapping from measurement variable name to IMG_DATA JP2 filepath.

        If ``prefer_res_m`` is not supplied, the variable's default/native
        resolution is used where possible.

        :param meas_vars:
            Requested measurement variable names. If omitted, default variables
            are used.
        :param prefer_res_m:
            Optional preferred resolution in metres. This acts as an override
            where that resolution exists for a variable.
        :return:
            Mapping ``{meas_var: filepath}``.
        :raises S2LayoutError:
            If any requested variable is not present.
        """
        if meas_vars is None:
            meas_vars = self.default_meas_vars()

        index: Dict[str, List[Path]] = {}
        for f in self.jp2_files():
            token = self._token_from_jp2_name(f.name)
            if token is None:
                continue
            index.setdefault(token, []).append(f)

        out: Dict[str, str] = {}
        missing: List[str] = []

        for mv in meas_vars:
            mv_norm = mv.upper()
            if re.fullmatch(r"\d{1,2}A?", mv_norm):
                mv_norm = "B" + mv_norm

            cands = index.get(mv_norm, [])
            if not cands:
                missing.append(mv)
                continue

            chosen = self._select_img_path_for_token(
                mv_norm,
                cands,
                prefer_res_m=prefer_res_m,
            )
            out[mv] = str(chosen)

        if missing:
            raise S2LayoutError(
                "Requested meas_vars not found in SAFE: "
                + ", ".join(missing)
                + f". Available: {', '.join(self.available_img_tokens())}"
            )

        return out

    def aux_data_dir(self, granule: Optional[Path] = None) -> Optional[Path]:
        """
        Return AUX_DATA directory if present.

        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            AUX_DATA directory path, or ``None`` if not found.
        """
        g = granule or self.granule_dir()
        cand = g / "AUX_DATA"
        if cand.exists():
            return cand
        deep = sorted(g.rglob("AUX_DATA"))
        return deep[0] if deep else None

    def aux_path(self, name: str, granule: Optional[Path] = None) -> Path:
        """
        Return a path to an AUX product.

        :param name:
            AUX product name.
        :param granule:
            Optional granule directory. If omitted, the primary granule is used.
        :return:
            Path to the AUX product.
        :raises S2LayoutError:
            If AUX_DATA is missing or the requested product cannot be found.
        """
        aux_dir = self.aux_data_dir(granule=granule)
        if aux_dir is None:
            raise S2LayoutError(f"AUX_DATA directory not found under: {self.granule_dir()}")

        cands = sorted(aux_dir.glob(name)) + sorted(aux_dir.glob(f"{name}*"))
        if cands:
            return cands[0]

        deep = sorted(aux_dir.rglob(name)) + sorted(aux_dir.rglob(f"{name}*"))
        if deep:
            return deep[0]

        raise S2LayoutError(f"AUX product '{name}' not found under {aux_dir}")


if __name__ == "__main__":
    pass
