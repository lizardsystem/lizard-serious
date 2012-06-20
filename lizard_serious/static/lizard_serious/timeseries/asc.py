#!/usr/bin/python
# -*- coding: utf-8 -*-
#***********************************************************************
#
# This file is part of the nens library.
#
# the nens library is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# the nens library is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the nens libraray.  If not, see
# <http://www.gnu.org/licenses/>.
#
# Copyright 2008, 2009 Mario Frasca
#*
#***********************************************************************
#* Library    : defines grids
#*
#* Project    : various
#*
#* $Id$
#*
#* initial programmer :  Mario Frasca
#* initial date       :  2008-07-30
#**********************************************************************

__revision__ = "$Rev$"[6:-2]

import logging
log = logging.getLogger('nens.asc')

import types
import numpy

gdal = None
try:
    from osgeo import gdal
except ImportError:
    pass


def name_to_location_name(name):
    if isinstance(name, types.ListType):
        name = '/'.join(name)
    name = name.replace('\\', '/')
    location = 'grid/'
    if name.find('/') != -1:
        parts = name.split('/')
        location = '/'.join(parts[:-1]) + '/'
        name = parts[-1]
    if location == '/':
        location = ''
    return location, name


def valid_float(v, nodata_value=None):
    """v or None

    >>> valid_float(0)
    0.0
    >>> print valid_float(12.3)
    12.3
    >>> print valid_float('4.32')
    4.32
    >>> print valid_float('4,32')
    4.32
    >>> print valid_float('')
    None
    >>> print valid_float('4.32', 4.32)
    4.32
    >>> print valid_float(-1000, -999)
    -999
    """

    if isinstance(v, str):
        v = v.replace(',', '.')
    try:
        v = float(v)
    except AttributeError:
        # in case the value was masked
        return nodata_value
    except ValueError:
        # the received string is invalid
        return nodata_value
    if v <= -999:
        return nodata_value
    if v == nodata_value:
        return nodata_value
    return v


def formatfloat(value, digits=3, nodata_value=-999):
    """same as %f, but avoids trailing zeroes

    >>> formatfloat(0)
    '0'
    >>> formatfloat(100)
    '100'
    >>> formatfloat(1.1)
    '1.1'
    >>> formatfloat(12.0004)
    '12'
    >>> formatfloat(12.0004, 4)
    '12.0004'
    >>> formatfloat(12.0004, 5)
    '12.0004'
    >>> formatfloat(numpy.ma.masked)
    '-999'
    >>> formatfloat(numpy.nan)
    '-999'
    """

    try:
        if numpy.ma.is_masked(value) or value != value:
            # nan (or masked in numpy 1.5++)
            value = nodata_value
    except AttributeError:
        # masked before numpy 1.5
        value = nodata_value
    result = "%0.8f" % round(value, digits)
    trim = 0
    while result[-trim - 1] == '0':
        trim += 1
    if trim:
        if result[-trim - 1] == '.':
            trim += 1
        result = result[:-trim]
    return result


import re
import os
import os.path


class AscGrid:

    whitespace = re.compile(r'[ \t]+')

    @classmethod
    def apply(cls, f, *grids):
        o = grids[0]
        ncols, nrows = o.ncols, o.nrows
        result = AscGrid(o.ncols, o.nrows, o.xllcorner, o.yllcorner, o.cellsize, nodata_value=o.nodata_value)
        result.values = f(*[g.values for g in grids])
        return result

    @classmethod
    def firstTimestampWithValue(cls, data, threshold=0.0, default_grid=None):
        """reads a .inc file and returns two AscGrid objects

        pixels in the first returned object hold the timestamp of the
        grid in the timestamped sequence of grids where the pixel
        reaches or exceeds the threshold

        pixels in the second returned object hold the first exceeding
        value
        """

        objects = cls.listFromStream(data, threshold=threshold, default_grid=default_grid)

        template, objects = objects[0], objects[1:]
        result_0 = template.copy()
        result_1 = template.copy()

        for timestamp, values_indexes in objects:
            for value, indexes in values_indexes.items():
                for col, row in indexes:
                    result_0[col, row] = timestamp
                    result_1[col, row] = value

        return result_0, result_1

    @classmethod
    def firstTimestamp(cls, data, default_value=None, default_grid=None,
                       threshold=True):
        """reads .inc file and returns list of grids of timestamps.

        'data' is a .inc file, defining timestamped grids, values in
        the grid are from the classes listed in the header of 'data'.

        firstTimestamp examines the list of timestamped grids
        represented by 'data' and returns a list containing one grid
        per class value.  each pixel of the grid associated to a class
        contains the first timestamp for which that pixel assumes the
        value of that class.  if a class value is never assumed at a
        pixel, in the corresponding grid the pixel contains a None.
        """

        objects = cls.listFromStream(data, default_value=default_value, default_grid=default_grid,
                                     threshold=threshold)

        template, objects = objects[0], objects[1:]
        result = {}

        for timestamp, values_indexes in objects:
            for value, indexes in values_indexes.items():
                for col, row in indexes:
                    if value not in result:
                        result[value] = template.copy()
                    result[value][col, row] = timestamp

        return sorted(result.items())

    @classmethod
    def maxIncrementsFromStream(cls, data, default_value=None, default_grid=None, end_in_constant=False, oneperhour=False, pertimeunit=False):
        """reads a .inc file and produces one AscGrid object.

        pixel per pixel, the returned object contains the maximum
        increment for that pixel on all subsequent AscGrids
        represented by the .inc file.
        """

        objects = cls.listFromStream(data, default_value=default_value, default_grid=default_grid, oneperhour=oneperhour)
        if end_in_constant:
            objects.append(objects[-1])
        o = objects[0][1]

        result = AscGrid(o.ncols, o.nrows, o.xllcorner, o.yllcorner, o.cellsize, nodata_value=o.nodata_value, default_value=default_value)
        for (time_im1, obj_im1), (time_i, obj_i) in zip(objects, objects[1:]):
            for row in range(1, o.nrows + 1):
                for col in range(1, o.ncols + 1):
                    previous, current = obj_im1[col, row], obj_i[col, row]
                    if current is None:
                        continue
                    if previous is None:
                        previous = 0.0
                    if result[col, row] is None or (current - previous > result[col, row]):
                        difference = float(current - previous)
                        if pertimeunit:
                            difference /= time_i - time_im1
                        result[col, row] = difference
        return result

    @classmethod
    def listFromStream(cls, data, output=None, namepattern=None,
                       oneperhour=False, default_value=None, just_count=False,
                       default_grid=None, threshold=None):
        """reads a .inc file and produces a list of pairs (timestamp, AscGrid object).

        this is the list version of xlistFromStream and relies on
        xlistFromStream to do its work.

        listFromStream accepts the same parameters as xlistFromStream,
        all with the same meanings except 'just_count'.  in addition,
        listFromStream accepts 'output' and 'namepattern'.

        if 'output' is specified, it must be a ZipFile object to which
        files will be saved.  'namepattern' is the name of the i-th
        file saved to output and may not be None.  In this case the
        function returns the estimated time_step and the amount of
        files written.

        if 'just_count' != False, the result is the length of the list
        generated.
        """

        generator = cls.xlistFromStream(data, oneperhour, default_value, just_count, default_grid, threshold=threshold)

        if just_count:
            return len([i for i in generator])

        if output is not None:
            time_step = last_timestamp = None
            written = 0
            for timestamp, item in generator:
                if time_step is None and last_timestamp is not None:
                    time_step = timestamp - last_timestamp
                last_timestamp = timestamp

                item.writeToStream(output, (namepattern or '%04d.asc') % written)
                written += 1

            return time_step, written

        return [i for i in generator]

    @classmethod
    def xlistFromStream(cls, data, oneperhour=False, default_value=None,
                        just_count=False, default_grid=None, threshold=None):
        """reads a .inc file and produces a generator of timestamped objects.

        if 'just_count' is False, each generated element is a pair
        where the first element is the hour of the file and the second
        element is the AscGrid object.

        if 'just_count' is not False, the timestamped objects are all
        the default_value, unmodified.

        if 'threshold' is not None, the first element of the list is
        not a pair but a single AscGrid matching the input data shape
        and the remainder of the returned generator holds
        dictionaries, described differently depending on the type of
        'threshold':

        if 'threshold' is a number, the generated dictionaries
        associate a value above the threshold to the pixels that
        exceed the threshold for that timestamp.  pixel coordinates
        appear only the first time they exceed the threshold.

        if 'threshold' is a boolean, the generated dictionaries
        associate class indexes to the list of coordinates of all
        pixels that take on the value from that class.  if 'threshold'
        is True, pixel coordinates appear only the first time they
        take on each class value.
        """

        import re
        splitter = re.compile(r'[^0-9\.\-]+')

        if threshold is not None and just_count:
            raise RuntimeError("can't just_count and use a threshold at the same time")

        def number(s):
            """convert string s to int or float, the one that fits
            """
            s = s.replace(',', '.')
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return None

        def get_number_parts(data, guard=None):
            i = data.readline().strip()
            if guard and i.startswith(guard):
                raise StopIteration()
            groups = splitter.split(i.strip())
            return [number(a) for a in groups]

        if isinstance(data, types.StringTypes):
            data = file(data)

        line = ''
        while not line.startswith("MAIN DIMENSIONS"):
            line = data.readline().strip()
            if line.lower().startswith('nodata_value'):
                # we really got a single grid
                if just_count:
                    obj = None
                else:
                    data.seek(0)
                    obj = AscGrid(data)
                log.debug("yielding object for fictive timestamp 0")
                yield (0, obj)
                raise StopIteration
        ncols, nrows = get_number_parts(data)
        if default_grid is not None:
            if ncols is None:
                ncols = default_grid.ncols
            if nrows is None:
                nrows = default_grid.nrows

        while not line.startswith("GRID"):
            line = data.readline().strip()
        cellfields = get_number_parts(data)
        try:
            # it usually is a .inc file
            xcellsize, ycellsize, xllcorner, yllcorner = cellfields
            start_of_grid = re.compile(r'^[ ]*-?[0-9]*\.[0-9]+ +[0-9]+ +[0-9]+ +[0-9]+ *$')
        except ValueError:
            # .fls files have a different structure
            cellsize, xllcorner, yllcorner = cellfields
            ycellsize = xcellsize = cellsize
            start_of_grid = re.compile(r'^[ ]*-?[0-9]*\.[0-9]+ +[0-9]+ +[0-9]+ *$')

        if xcellsize != ycellsize:
            raise ValueError("AscGrid can't cope with rectangular cells")

        xllcorner -= xcellsize / 2
        yllcorner -= ycellsize / 2
        while not line.startswith("CLASSES OF INCREMENTAL FILE"):
            line = data.readline().strip()

        classes = []
        while True:
            try:
                class_def = get_number_parts(data, "ENDCLASSES")
                classes.append(class_def)
            except StopIteration:
                break

        obj = default_value
        if threshold is not None:
            assigned = set()
            yield AscGrid(ncols, nrows, xllcorner, yllcorner, xcellsize, -999, default_value=None)
            obj = {}
        obj_is_to_yield = False
        yielded_hour = None
        hour = None

        while True:
            line = data.readline().strip()
            if not line:
                if obj_is_to_yield:
                    log.debug("yielding object for timestamp %s" % hour)
                    yield (hour, obj)
                raise StopIteration
            if start_of_grid.match(line):
                if obj_is_to_yield:
                    log.debug("yielding object for timestamp %s" % hour)
                    if threshold is None or obj != {}:
                        yield (hour, obj)
                    yielded_hour = int(hour)
                hour, _, class_column = splitter.split(line.strip())[:3]
                hour = float(hour)
                class_column = int(class_column)
                obj_is_to_yield = (not oneperhour or yielded_hour != int(hour))
                if threshold is not None:
                    obj = {}
                elif not just_count:
                    obj = AscGrid(ncols, nrows, xllcorner, yllcorner, xcellsize,
                                  nodata_value=-999, default_value=obj)
                continue
            if not just_count:
                groups = splitter.split(line.strip())
                col, row, class_no = [number(a) for a in groups]
                if threshold is not None:
                    if class_no == 0:
                        value = 0.0
                    else:
                        value = classes[class_no - 1][class_column - 1]
                    if (value >= threshold > 0 or value > threshold >= 0) or isinstance(threshold, bool):
                        if isinstance(threshold, bool):
                            check = (class_no, col, row)
                        else:
                            check = (col, row)
                        if check not in assigned:
                            if threshold is not False:
                                assigned.add(check)
                            obj.setdefault(value, [])
                            obj[value].append((col, (nrows - row + 1)))
                elif class_no is not 0:
                    obj[col, (nrows - row + 1)] = classes[class_no - 1][class_column - 1]
                else:
                    obj[col, (nrows - row + 1)] = 0.0

    def copy(self):
        """returns a new object equal to self"""

        return AscGrid(self.ncols, self.nrows, self.xllcorner, self.yllcorner,
                       self.cellsize, nodata_value=self.nodata_value, default_value=None)

    def __init__(self, *args, **kwargs):
        if (1 <= len(args) <= 2 and kwargs == {}) or 'data' in kwargs.keys():
            self._init_from_data(*args, **kwargs)
        else:
            self._init_from_scratch(*args, **kwargs)
        self.images = {}

    def _init_from_data(self, data, name=None, cellsize=None, xllcorner=None, yllcorner=None, nodata_value=-999.0):
        self.location = self.srcname = ''
        self.source = None
        read_params = []
        if hasattr(data, "GetGeoTransform"):
            import struct
            geotransform = data.GetGeoTransform()
            self.nrows = data.RasterYSize
            self.ncols = data.RasterXSize
            self.xllcorner = geotransform[0]
            self.yllcorner = geotransform[3] - geotransform[1] * self.nrows
            self.cellsize = geotransform[1]
            self.nodata_value = nodata_value
            description = data.GetDescription()
            self.source = description + '.asc'
            self.location = os.path.dirname(description)
            self.srcname = os.path.basename(description)

            band = data.GetRasterBand(1)
            dataType = band.DataType
            unpacking_types = [(None, None),
                               ('c', numpy.ubyte),
                               ('H', numpy.uint16),
                               ('h', numpy.int16),
                               ('I', numpy.uint32),
                               ('i', numpy.int32),
                               ('f', numpy.float32),
                               ('d', numpy.float64)]
            unpack_char, numpy_type = unpacking_types[dataType]
            content = numpy.ndarray((band.YSize, band.XSize), dtype=numpy_type)
            content[:] = 0
            for y in range(band.YSize):
                scanline = band.ReadRaster(0, y, band.XSize, 1, band.XSize, 1, dataType)
                tuple_of_values = struct.unpack(unpack_char * band.XSize, scanline)
                content[y, :] = tuple_of_values
            self.values = numpy.ma.masked_values(content, band.GetNoDataValue())

            ## DONE
            return

        elif isinstance(data, numpy.ndarray):
            log.debug("initializing AscGrid from numpy.ndarray" % data)
            if xllcorner is None or yllcorner is None or cellsize is None:
                raise RuntimeError("initialization from numpy.ndarray needs metadata.")
            self.xllcorner = xllcorner
            self.yllcorner = yllcorner
            self.cellsize = cellsize
            self.nodata_value = nodata_value
            self.location = 'grid/'
            self.srcname = ''

            self.values = data.copy
            self.nrows, self.ncols = data.shape

            ## DONE
            return

        elif isinstance(data, types.StringTypes):
            log.debug("initializing AscGrid from '%s'" % data)
            self.location, self.srcname = name_to_location_name(data)
            data = file(self.location + self.srcname)
        elif name is not None:
            log.debug("initializing AscGrid from zipfile")
            read_params = [name]
        else:
            log.debug("initializing AscGrid from data stream")

        ## if we got here: data is more or less a stream
        data = data.read(*read_params)
        data = [i.strip() for i in data.split('\n') if not i.strip().startswith("/*")]

        header_len = 0
        for header_line in data:
            fields = self.whitespace.split(header_line)
            if len(fields) > 2:
                break
            self.__dict__[fields[0].lower()] = float(fields[1].replace(',', '.'))
            header_len += 1

        self.nrows = int(self.nrows)
        self.ncols = int(self.ncols)

        content = numpy.zeros((self.nrows, self.ncols))
        for rowno, line in enumerate(data[header_len:]):
            values = [valid_float(i, self.nodata_value) for i in self.whitespace.split(line) if i]
            if values:
                content[rowno, :] = values
        self.values = numpy.ma.masked_values(content, self.nodata_value)

    def _init_from_scratch(self, ncols, nrows, xllcorner, yllcorner,
                            cellsize, nodata_value, default_value=0.0):
        log.debug("initializing AscGrid from parameters")
        self.ncols = int(ncols)
        self.nrows = int(nrows)
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.cellsize = cellsize
        self.nodata_value = nodata_value
        self.location = 'grid/'
        self.srcname = ''
        content = numpy.zeros((self.nrows, self.ncols))
        if default_value == nodata_value:
            default_value = None
        for row in range(nrows):
            if isinstance(default_value, AscGrid):
                values = default_value.values.filled(nodata_value)[row, :]
            else:
                values = default_value

            content[row, :] = values
        self.values = numpy.ma.masked_values(content, nodata_value)

    def _pair_to_coords(self, pair):
        try:
            col, row = pair
            get_by_rowcol = (1 <= col <= self.ncols) and (1 <= row <= self.nrows)
        except (TypeError, AttributeError):
            pair = pair.x, pair.y
            get_by_rowcol = False
        if get_by_rowcol:
            col -= 1
            row -= 1
        else:
            x, y = pair
            col = int((x - self.xllcorner) / self.cellsize)
            row = int(self.nrows - int(y - self.yllcorner) / self.cellsize)
        return col, row

    def __getitem__(self, pair, value=None):
        col, row = self._pair_to_coords(pair)
        try:
            if col < 0 or row < 0:
                return False
            result = self.values[row, col]
            try:
                if result == self.nodata_value:
                    return None
                if result is numpy.ma.masked:
                    return None
                if result != result:
                    return None
            except:
                # checking fails prior to 1.5.1 if element is masked
                return None
            return result
        except IndexError:
            return False

    def __setitem__(self, pair, value):
        col, row = self._pair_to_coords(pair)
        if value == self.nodata_value:
            value = None
        self.values[row, col] = value

    def scoreatpercentile(self, per):
        try:
            from scipy.stats import scoreatpercentile
        except ImportError:
            log.error("can't import scoreatpercentile from scipy.stats")
            return None
        return scoreatpercentile(self.values.compressed(), per)

    def save(self, destdir=None):
        if destdir is None:
            dest = self.source
        else:
            destdir = destdir.replace('\\', '/')
            if not destdir.endswith('/'):
                destdir += '/'
            log.debug("writing AscGrid to '%s' destination directory" % destdir)
            location, name = name_to_location_name(self.srcname)
            dest = destdir + name
        output = file(dest, "w")
        self.writeToStream(output)
        output.close()

    def headerLines(self):
        """the list of lines for the .asc header

        >>> obj = AscGrid(5, 5, xllcorner=15.0, yllcorner=12.5, cellsize=0.5, nodata_value=-999)
        >>> obj.headerLines()
        ['nCols        5', 'nRows        5', 'xllCorner    15', 'yllCorner    12.5', 'CellSize     0.5', 'nodata_value -999']
        """

        result = []
        for fieldname in ['nCols', 'nRows']:
            line = ("%-13s" % fieldname)
            line += "%i" % getattr(self, fieldname.lower())
            result.append(line)

        for fieldname in ['xllCorner', 'yllCorner', 'CellSize', 'nodata_value']:
            line = ("%-13s" % fieldname)
            line += formatfloat(getattr(self, fieldname.lower()),
                                nodata_value=self.nodata_value)
            result.append(line)

        return result

    def writeToStream(self, output, name=None):
        "writes self to a stream or a zipfile"

        log.debug("about to save self to %s(%s)" % (type(output), name))
        result = self.headerLines()
        for row in self.values.filled(self.nodata_value):
            stringvalues = [formatfloat(value, nodata_value=self.nodata_value)
                            for value in row]
            result.append(' ' + ' '.join(stringvalues))

        result.append('')
        result = '\n'.join(result)

        output_methods = output.__class__.__dict__
        try:
            temp = result.replace('\n', '\r\n')
            if name is None:
                location, name = self.location, self.srcname
            else:
                location, name = name_to_location_name(name)
                if not name:
                    name = self.srcname
            output.writestr(str(location + name), temp)
            log.info("wrote item '%s' to zipfile" % (location + name))
            return
        except AttributeError:
            pass

        try:
            output.write(result)
            log.debug("writing item file")
            return
        except AttributeError:
            pass

        raise ValueError("don't know how to write to a %(__name__)s" % output_methods)

    def get_col_row(self, pair):
        col, row = self._pair_to_coords(pair)
        return (col + 1, row + 1)
        pass

    def point(self, pair, percent=(0.5, 0.5)):
        col, row = self._pair_to_coords(pair)
        if not (0 <= row < self.nrows) or not (0 <= col <= self.ncols):
            raise ValueError("point falls outside of the grid")
        x = self.xllcorner + self.cellsize * (col + percent[0])
        y = self.yllcorner + self.cellsize * ((self.nrows - row - 1) + percent[1])
        return (x, y)

    def asImage(self, colorMapping, bits=24):
        """calculates an image that represents the current object given the
        colorMapping.

        colorMapping: anything with method getColor: float->(r, g, b).
        bits: either 8 (palette, black for transparency) or 24 (RGB+alpha).
        """

        from PIL import Image
        if bits == 8:
            palette = [(0, 0, 0)]
            for boundary, color in colorMapping.values:
                palette.append(color)
            result = Image.new(size=(self.ncols, self.nrows), mode='P', color=0)
            p = []
            for r, g, b in palette:
                p.extend([r, g, b])
            result.putpalette(p)
        elif bits == 24:
            result = Image.new(size=(self.ncols, self.nrows), mode='RGBA')
        else:
            raise ValueError("bits must be either 8 or 24")

        if (bits) not in self.images:
            data = []
            for row in self.values.filled(0):
                for cell in row:
                    color = colorMapping.getColor(cell)
                    if bits == 8:
                        color = palette.index(color)
                    else:
                        if color == (0, 0, 0):
                            color = (0, 0, 0, 0)
                    data.append(color)

            self.images[bits] = data
        result.putdata(self.images[bits])
        return result


class ColorMapping:

    def __init__(self, data_stream):
        """create a ColorMapping object

        reads the color mapping from the data_stream.
        """

        header = data_stream.readline()
        fields = [i.lower().strip() for i in header.split(',')]
        if fields != ['leftbound', 'colour']:
            raise Exception("unexpected mapping file format")

        self.values = []
        while True:
            line = data_stream.readline()
            if not line:
                break
            fields = [i.lower().strip() for i in line.split(',')]
            if not fields:
                continue
            bound = float(fields[0])
            color = [int(fields[1][i * 2:(i + 1) * 2], 16) for i in range(3)]
            self.values.append((bound, tuple(color)))

        self.palette = None

    def getColor(self, value):
        """returns color associated to value

        either black or the color associated to the greatest value
        that is smaller than the given one"""

        if value is None:
            return (0, 0, 0)
        try:
            found = max([(v, c) for (v, c) in self.values if v < value])
            return found[1]
        except ValueError:
            return (0, 0, 0)

    def getPaletteIndex(self, value):
        "returns the index of the color associated to value"

        # make sure that palette has been calculated
        self.asPalette()

        return None

    def asPalette(self):
        "returns the palette associated to the color mapping"

        if self.palette is None:
            pass

        return self.palette
