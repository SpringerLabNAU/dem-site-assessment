# dem_mfr/main.py
from __future__ import annotations
import os, zipfile, requests
import geopandas as gpd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import generic_filter, uniform_filter, distance_transform_edt
from scipy.signal import convolve2d
from skimage.morphology import disk

TNM_SEARCH_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

@dataclass
class TNMProduct:
    id: str
    title: str
    downloadURL: str
    format: str

@dataclass
class WorkflowConfig:
    kernel_radius_m: float = 300.0
    target_epsg: Optional[int] = 26912

def load_aoi(path: str) -> gpd.GeoDataFrame:
    """
    Load an AOI from:
      - GeoJSON / JSON,
      - Shapefile directory (.shp in a folder),
      - Zipped shapefile (.zip)
    Returns GeoDataFrame in EPSG:4326.
    """
    p = path.lower()
    if p.endswith(".geojson") or p.endswith(".json"):
        gdf = gpd.read_file(path)
    elif p.endswith(".zip"):
        # zipped shapefile
        gdf = gpd.read_file(f"zip://{os.path.abspath(path)}")
    else:
        # assume folder or .shp path
        gdf = gpd.read_file(path)
    return gdf.to_crs(4326)

def extract_tilecode(title: str) -> str | None:
    """
    Extracts a tile code like 'n35w111' or 'n34w110' from a TNM title.
    Returns lowercase code or None if not found.
    """
    if not title:
        return None
    m = re.search(r'\b([ns]\d{2}[ew]\d{3})\b', title.lower())
    return m.group(1) if m else None

def tnm_search_dem(
    aoi_4326,
    resolution_like: Optional[str] = None,    # e.g., "1 arc-second", "1/3 arc-second", "1 meter"
    max_results: int = 200,
    prefer_geotiff: bool = True,
    category: str = "Elevation",
    title_must_include: str =None,          # keep it generic; matches “USGS 1 arc-second DEM…”
) -> List[TNMProduct]:
    """
    Robust TNM search: query by bbox (+ category/prodFormats), then filter title client-side.
    """
    minx, miny, maxx, maxy = aoi_4326.total_bounds
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "max": max_results,
        "category": category,
    }
    if prefer_geotiff:
        params["prodFormats"] = "GeoTIFF"

    r = requests.get(TNM_SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    items = r.json().get("items", [])

    # Build a regex for the resolution if provided
    res_ok = lambda _t: True
    if resolution_like:
        rl = resolution_like.strip().lower()
        # normalize a couple of common variants
        rl = rl.replace("arcsecond", "arc-second").replace("arc second", "arc-second")
        pat = rl.replace("1/3", r"1/?3").replace(" ", r"\s*")
        res_re = re.compile(pat)
        def res_ok(_t): return bool(res_re.search(_t))

    title_need = (title_must_include or "").lower()

    out: List[TNMProduct] = []
    for it in items:
        title = (it.get("title") or "")
        t = title.lower()
        if title_need and title_need not in t:
            continue
        if not res_ok(t):
            continue
        out.append(TNMProduct(
            id=str(it.get("id","")),
            title=title,
            downloadURL=str(it.get("downloadURL") or ""),
            format=str(it.get("format","")),
        ))
    return out

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_existing_tif(out_dir: str | Path, tilecode: str | None) -> str | None:
    """
    If a GeoTIFF already exists in out_dir (optionally matching tilecode), return it.
    """
    out_dir = Path(out_dir)
    if tilecode:
        for t in out_dir.glob(f"*{tilecode}*.tif"):
            return str(t)
    # fallback: any tif
    any_tif = next(out_dir.glob("*.tif"), None)
    return str(any_tif) if any_tif else None

def download_first_geotiff_if_needed(prod: "TNMProduct", out_dir: str | Path, overwrite: bool=False) -> str | None:
    """
    Idempotent download:
      - If a matching tile .tif is already in out_dir, return it.
      - Else download (ZIP or TIF), extract if needed, and return the .tif path.
    """
    out_dir = ensure_dir(out_dir)
    tilecode = extract_tilecode(prod.title)

    if not overwrite:
        existing = find_existing_tif(out_dir, tilecode)
        if existing:
            return existing

    if not prod.downloadURL:
        return None

    dst = out_dir / os.path.basename(prod.downloadURL.split("?")[0])

    # Download artifact (zip or tif)
    with requests.get(prod.downloadURL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)

    # If it’s already a .tif
    if dst.suffix.lower() == ".tif":
        return str(dst)

    # If it’s a ZIP, extract first .tif
    if dst.suffix.lower() == ".zip":
        with zipfile.ZipFile(dst) as z:
            tifs = [m for m in z.namelist() if m.lower().endswith(".tif")]
            if not tifs:
                return None
            z.extract(tifs[0], out_dir)
            return str(out_dir / tifs[0])

    return None


def get_epsg_code(crs) -> Optional[int]:
    """Return EPSG integer if available."""
    try:
        return crs.to_epsg()
    except Exception:
        return None

def reproject_to_epsg(src_tif: str, target_epsg: int,
                      resampling: Resampling = Resampling.bilinear,
                      overwrite: bool=False) -> str:
    """
    Reproject a GeoTIFF to target EPSG. If the output already exists and overwrite=False, return it.
    """
    base, ext = os.path.splitext(src_tif)
    out = f"{base}_EPSG{target_epsg}.tif"
    if not overwrite and Path(out).exists():
        return out

    with rasterio.open(src_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{target_epsg}", src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(
            crs=f"EPSG:{target_epsg}",
            transform=transform,
            width=width,
            height=height,
            nodata=np.nan,
            count=1,
            dtype=rasterio.float32,
        )
        with rasterio.open(out, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=f"EPSG:{target_epsg}",
                resampling=resampling,
                dst_nodata=np.nan,
            )
    return out

def ensure_projection(src_tif: str, target_epsg: int | None, overwrite: bool=False) -> str:
    """
    If target_epsg is set and source isn't already that EPSG, reproject (idempotent).
    Otherwise return src_tif.
    """
    if not target_epsg:
        return src_tif
    with rasterio.open(src_tif) as src:
        cur = src.crs.to_epsg() if src.crs else None
    if cur == target_epsg:
        return src_tif
    return reproject_to_epsg(src_tif, target_epsg, overwrite=overwrite)

def read_raster(path: str) -> Tuple[np.ndarray, Affine, dict]:
    """
    Read a single-band GeoTIFF as float; convert nodata to NaN.
    Returns: (array, transform, meta)
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.meta.copy()

def write_raster(path: str, arr: np.ndarray, meta: dict) -> None:
    """
    Write a single-band GeoTIFF (float32, NaN as nodata).
    """
    meta2 = meta.copy()
    meta2.update(count=1, dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(arr.astype(np.float32), 1)

def meters_to_pixels(radius_m: float, transform: Affine) -> int:
    """
    Convert a radius in meters to pixels using |transform.a| as pixel width.
    Assumes square pixels post-reprojection.
    """
    px = int(np.ceil(radius_m / abs(transform.a)))
    return max(px, 1)

def circular_kernel(radius_px: int) -> np.ndarray:
    """
    Binary circular kernel (True/1 inside circle) with radius in pixels.
    """
    return disk(radius_px)

def focal_mean(arr: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """
    NaN-safe mean over footprint. Centers align; edges padded with NaN.
    """
    pad_y, pad_x = footprint.shape[0]//2, footprint.shape[1]//2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant", constant_values=np.nan)
    fp_flat = footprint.flatten().astype(bool)

    def func(vals):
        v = vals[fp_flat]
        v = v[~np.isnan(v)]
        return np.nan if v.size == 0 else np.nanmean(v)

    out = generic_filter(padded, func, size=footprint.shape,
                         mode="constant", cval=np.nan)
    return out[pad_y:-pad_y, pad_x:-pad_x]

## function to calculate TPI

def tpi(arr: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """
    Topographic Position Index (Weiss, 2001): elevation - neighborhood mean.
    """
    return arr - focal_mean(arr, footprint)

## functions to calculate terrain roughness index

def tri(arr: np.ndarray, transform, radius_m: float = 30.0) -> np.ndarray:
    """
    Terrain Ruggedness Index (Riley 1999) with adjustable neighborhood radius.
    radius_m is converted to pixels using transform.a (assumes UTM / meters).

    TRI = sqrt( sum( (neighbor - center)^2 ) / N ), NaN-safe.
    """
    r_px = meters_to_pixels(radius_m, transform)
    footprint = circular_kernel(r_px)
    fp_flat = footprint.flatten().astype(bool)

    def f(window):
        c = window[len(window) // 2]
        if np.isnan(c):
            return np.nan
        diffs = window[fp_flat] - c
        diffs = diffs[~np.isnan(diffs)]
        return np.sqrt(np.nansum(diffs ** 2))

    pad_y, pad_x = footprint.shape[0] // 2, footprint.shape[1] // 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant", constant_values=np.nan)
    out = generic_filter(padded, f, size=footprint.shape,
                         mode="constant", cval=np.nan)
    return out[pad_y:-pad_y, pad_x:-pad_x]

## Functions to calculate Vector Ruggedness--------------------------------------
def nanfill_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaNs by copying the nearest valid cell (fast EDT-based)."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    # indices of nearest non-NaN cells
    _, inds = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = a.copy()
    filled[mask] = a[tuple(inds[:, mask])]
    return filled

def slope_horn(arr: np.ndarray, transform, units: str = "degrees", nan_fill: bool = True) -> np.ndarray:
    """
    Slope from a DEM using Horn (1981)-style 3x3 derivatives.
    units: "degrees" (default) or "percent"
    nan_fill: if True, fills DEM NaNs with nearest neighbor before gradients (like VRM)
    """
    a = nanfill_nearest(arr) if nan_fill else arr

    # pixel size (meters) from transform; handle negative e
    dx = abs(transform.a)
    dy = abs(transform.e)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float) / (8 * dx)
    ky = np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]], dtype=float) / (8 * dy)

    dzdx = convolve2d(a, kx, mode="same", boundary="symm")
    dzdy = convolve2d(a, ky, mode="same", boundary="symm")

    # slope (radians)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))

    if units.lower().startswith("deg"):
        out = np.degrees(slope_rad)
    elif units.lower().startswith("per"):
        out = np.tan(slope_rad) * 100.0
    else:
        raise ValueError("units must be 'degrees' or 'percent'")

    # restore NaNs where input was NaN
    out[np.isnan(arr)] = np.nan
    return out

def _horn_slopes(arr: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate slope & aspect (radians) using Horn 3x3 kernels.
    transform.a ~ pixel width (m), transform.e ~ pixel height (m, negative).
    """
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float) / (8 * abs(transform.a))
    ky = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], float) / (8 * abs(transform.e))
    dzdx = convolve2d(arr, kx, mode="same", boundary="symm")
    dzdy = convolve2d(arr, ky, mode="same", boundary="symm")
    slope = np.arctan(np.hypot(dzdx, dzdy))          # radians
    aspect = np.arctan2(dzdy, -dzdx)                 # radians, 0 = east, CCW
    return slope, aspect

def vrm(arr: np.ndarray,
        transform,
        radius_m: float | None = None,
        smooth_px: int = 3,
        nan_fill: bool = True) -> np.ndarray:
    """
    Vector Ruggedness Measure (Sappington 2007), NaN-robust.

    - If nan_fill=True, fill NaNs in DEM by nearest neighbor before gradients.
    - Neighborhood can be set by:
        * radius_m (meters)  -> circular ~size via converted pixel span, applied as square smoothing of unit vectors
        * smooth_px (odd int)-> fallback square window size
    """
    # Fill NaNs to avoid convolution poisoning
    a = nanfill_nearest(arr) if nan_fill else arr

    # Horn slopes/aspect
    slope, aspect = _horn_slopes(a, transform)

    # Unit vectors
    x = np.sin(slope) * np.cos(aspect)
    y = np.sin(slope) * np.sin(aspect)
    z = np.cos(slope)

    # Choose smoothing window
    if radius_m is not None:
        r_px = meters_to_pixels(radius_m, transform)
        smooth_px = max(3, 2 * int(r_px) + 1)  # odd, >=3

    # Smooth unit vectors (windowed mean)
    Rx = uniform_filter(x, size=smooth_px)
    Ry = uniform_filter(y, size=smooth_px)
    Rz = uniform_filter(z, size=smooth_px)

    R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    return 1.0 - R

# --- Binary reclass, histograms, and rule combination -----------------------


def finite_values(arr: np.ndarray) -> np.ndarray:
    """Return 1-D array of finite values (drop NaNs/±inf) for histograms."""
    return arr[np.isfinite(arr)]

def histogram(arr: np.ndarray, bins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-safe histogram: returns (counts, bin_edges)."""
    v = finite_values(arr)
    if v.size == 0:
        return np.array([0]), np.array([0, 1])
    counts, edges = np.histogram(v, bins=bins)
    return counts, edges

def binary_threshold(
    arr: np.ndarray,
    threshold: float,
    direction: str = ">",        # options: ">", ">=", "<", "<="
    true_class: int = 1,         # which class gets the value 1 (the other gets 0)
    nan_as: int = 0              # NaN handling in output: 0 or 1
) -> np.ndarray:
    """
    Return a 0/1 array from thresholding. NaNs mapped to nan_as.
    """
    a = np.asarray(arr)
    out = np.zeros_like(a, dtype=np.uint8)
    valid = np.isfinite(a)

    if direction == ">":
        mask = a > threshold
    elif direction == ">=":
        mask = a >= threshold
    elif direction == "<":
        mask = a < threshold
    elif direction == "<=":
        mask = a <= threshold
    else:
        raise ValueError("direction must be one of '>', '>=', '<', '<='")

    # map to classes
    if true_class == 1:
        out[mask & valid] = 1
        out[~valid] = nan_as
    else:
        # true condition gets 0, false gets 1
        out[~mask & valid] = 1
        out[~valid] = nan_as
    return out

def binary_range(
    arr: np.ndarray,
    min_val: float | None = None,
    max_val: float | None = None,
    inclusive: bool = True,
    true_class: int = 1,
    nan_as: int = 0
) -> np.ndarray:
    """
    0/1 mask where values fall within [min_val, max_val] (if inclusive) or (min_val, max_val).
    If min_val is None: only max cutoff is used; if max_val is None: only min cutoff is used.
    """
    a = np.asarray(arr)
    out = np.zeros_like(a, dtype=np.uint8)
    valid = np.isfinite(a)

    if min_val is None and max_val is None:
        raise ValueError("Provide at least min_val or max_val.")

    if inclusive:
        cond = np.ones_like(a, dtype=bool)
        if min_val is not None:
            cond &= (a >= min_val)
        if max_val is not None:
            cond &= (a <= max_val)
    else:
        cond = np.ones_like(a, dtype=bool)
        if min_val is not None:
            cond &= (a > min_val)
        if max_val is not None:
            cond &= (a < max_val)

    if true_class == 1:
        out[cond & valid] = 1
        out[~valid] = nan_as
    else:
        out[(~cond) & valid] = 1
        out[~valid] = nan_as
    return out

def boolean_and(*masks: np.ndarray) -> np.ndarray:
    """Logical AND across multiple 0/1 masks, returning 0/1 uint8."""
    if not masks:
        raise ValueError("At least one mask required.")
    out = (masks[0] > 0)
    for m in masks[1:]:
        out &= (m > 0)
    return out.astype(np.uint8)

def boolean_or(*masks: np.ndarray) -> np.ndarray:
    """Logical OR across multiple 0/1 masks, returning 0/1 uint8."""
    if not masks:
        raise ValueError("At least one mask required.")
    out = (masks[0] > 0)
    for m in masks[1:]:
        out |= (m > 0)
    return out.astype(np.uint8)

def composite_rules(
    layers: Dict[str, np.ndarray],
    rules: Dict[str, Tuple[str, float | Tuple[float, float], int]],
    combine: str = "AND"
) -> np.ndarray:
    """
    Apply per-layer (op, threshold_or_range, true_class) and combine.
    Supported ops:
      - '>', '>=', '<', '<='
      - 'abs>', 'abs<' (uses absolute value of the layer)
      - 'between' -> value is (min, max)
      - 'between_deg' / 'between_pct' (for slope arrays if you bake them as slope_deg or slope_pct)
    """
    masks = []
    for lname, (op, val, true_class) in rules.items():
        if lname not in layers:
            raise KeyError(f"Layer '{lname}' not found in provided layers.")
        arr = layers[lname]

        if op in (">", ">=", "<", "<="):
            thr = float(val)
            m = binary_threshold(arr, thr, direction=op, true_class=true_class, nan_as=0)

        elif op in ("abs>", "abs<"):
            thr = float(val)
            if op == "abs>":
                m = binary_threshold(np.abs(arr), thr, direction=">", true_class=true_class, nan_as=0)
            else:
                m = binary_threshold(np.abs(arr), thr, direction="<", true_class=true_class, nan_as=0)

        elif op in ("between", "between_deg", "between_pct"):
            if not isinstance(val, (tuple, list)) or len(val) != 2:
                raise ValueError(f"For '{op}', provide (min, max).")
            vmin, vmax = float(val[0]), float(val[1])
            m = binary_range(arr, vmin, vmax, inclusive=True, true_class=true_class, nan_as=0)

        else:
            raise ValueError(f"Unsupported operation '{op}'")

        masks.append(m)

    if combine.upper() == "AND":
        return boolean_and(*masks)
    elif combine.upper() == "OR":
        return boolean_or(*masks)
    else:
        raise ValueError("combine must be 'AND' or 'OR'")


def write_binary_raster(path: str, mask01: np.ndarray, meta: dict) -> None:
    """Write 0/1 mask as UInt8 (nodata=255 to keep 0 and 1 clean)."""
    m = meta.copy()
    m.update(count=1, dtype="uint8", nodata=255)
    import rasterio
    with rasterio.open(path, "w", **m) as dst:
        dst.write(mask01.astype(np.uint8), 1)


def classify_and_save(layers, rules, meta, out_raster_path: str, combine: str = "AND") -> np.ndarray:
    """
    Apply composite_rules, write 0/1 GeoTIFF, and return the mask.
    """
    mask = composite_rules(layers, rules, combine=combine)
    write_binary_raster(out_raster_path, mask, meta)
    return mask

def write_rules_txt(txt_path: str, rules: dict, notes: str | None = None) -> None:
    """
    Write a human-readable summary of the rules to a .txt file.
    """
    lines = []
    lines.append("Binary reclassification rules\n")
    if notes:
        lines.append(f"Notes: {notes}\n")
    lines.append("Legend: (op, value, true_class)\n")
    for k, v in rules.items():
        try:
            op, val, cls = v
        except Exception:
            lines.append(f"{k}: {v}  # (unexpected shape)\n")
            continue
        if op in ("between", "between_deg", "between_pct") and isinstance(val, (tuple, list)) and len(val) == 2:
            lines.append(f"{k}: {op} {val[0]}..{val[1]} → class {cls}\n")
        elif op in ("abs>", "abs<"):
            lines.append(f"{k}: {op} {val} → class {cls}\n")
        else:
            lines.append(f"{k}: {op} {val} → class {cls}\n")
    from datetime import datetime
    lines.append(f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)