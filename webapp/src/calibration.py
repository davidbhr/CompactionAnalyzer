"""Read explicit microscopy calibration; ordinary display DPI is not a pixel size."""
import math
import xml.etree.ElementTree as ET


def assumed_calibration(note='No microscopy calibration found.'):
    return dict(pixel_size_um=1., source='assumed', assumed=True,
                note=note + ' Default 1 µm/px; check this value.')


def _unit_um(unit):
    unit = str(unit).strip().lower().replace('µ', 'u').replace('μ', 'u')
    return {'um': 1., 'micron': 1., 'microns': 1., 'micrometer': 1.,
            'micrometers': 1., 'micrometre': 1., 'micrometres': 1., 'nm': .001,
            'mm': 1000., 'cm': 10000., 'm': 1000000.}[unit]


def _square_pixels(x, y, source):
    if not all(math.isfinite(v) and v > 0 for v in (x, y)):
        raise ValueError('Invalid pixel size')
    if not math.isclose(x, y, rel_tol=1e-4):
        return assumed_calibration('X/Y pixel sizes differ; resample to square pixels for physical units.')
    return dict(pixel_size_um=x, source=source, assumed=False,
                note=f'From {source}; editable.')


def tiff_calibration(handle):
    """OME physical sizes or ImageJ units plus X/Y resolution, in µm/pixel."""
    try:
        if handle.ome_metadata:
            xml = handle.ome_metadata
            if len(xml) > 1_000_000:
                return assumed_calibration('Microscopy metadata is too large to inspect.')
            pixels = next(node for node in ET.fromstring(xml).iter()
                          if node.tag.rsplit('}', 1)[-1] == 'Pixels')
            sizes = [float(pixels.attrib['PhysicalSize' + axis]) *
                     _unit_um(pixels.attrib.get('PhysicalSize' + axis + 'Unit', 'µm'))
                     for axis in ('X', 'Y')]
            return _square_pixels(*sizes, 'OME-TIFF')
        metadata = handle.imagej_metadata
        if metadata and 'unit' in metadata:
            unit = _unit_um(metadata['unit'])
            sizes = []
            for axis in ('X', 'Y'):
                numerator, denominator = handle.pages[0].tags[axis + 'Resolution'].value
                sizes.append(unit * denominator / numerator)
            return _square_pixels(*sizes, 'ImageJ TIFF')
    except (KeyError, TypeError, ValueError, ZeroDivisionError, StopIteration, ET.ParseError):
        return assumed_calibration('Microscopy calibration is incomplete or unsupported.')
    return assumed_calibration()


def sigma_in_pixels(request):
    sigma = float(request['sigma'])
    unit = request.get('sigma_unit', 'pixel')
    if unit == 'µm':
        try:
            size = float(request.get('calibration', {}).get('pixel_size_um', 0))
        except (TypeError, ValueError):
            raise ValueError('Enter a positive pixel size to use sigma in µm.') from None
        if not math.isfinite(size) or size <= 0:
            raise ValueError('Enter a positive pixel size to use sigma in µm.')
        sigma /= size
    elif unit != 'pixel':
        raise ValueError('Tensor window unit must be pixel or µm.')
    # Unit round trips may put a boundary one floating-point step outside it.
    if math.isclose(sigma, .1, rel_tol=1e-12):
        sigma = .1
    if math.isclose(sigma, 100, rel_tol=1e-12):
        sigma = 100.
    if not math.isfinite(sigma) or not .1 <= sigma <= 100:
        raise ValueError('Tensor window must correspond to 0.1–100 pixels.')
    return sigma
