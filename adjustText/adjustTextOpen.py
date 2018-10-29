from __future__ import division
import sys
#from matplotlib import pyplot as plt
from itertools import product
import numpy as np
from operator import itemgetter

""""""

if sys.version_info >= (3, 0):
    xrange = range

DEBUG = False
# DEBUG = True
def debugprint(*x, **kwargs):
    """Print various status and update messages. Makes up for the
    poor traceback available when running a server"""
    if DEBUG:
        print('***', *x, **kwargs)
    return


class Box:
    """BBox and bas for Ax class. """
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.width = xmax - xmin
        self.height = ymax - ymin
    # def transData(self, x, y):
    #     return (x - self.xmin) / self.xspan, (y - self.ymin) / self.yspan
    def __repr__(self):
        return "Box({}, {}, {}, {})".format(self.xmin, self.xmax, self.ymin, self.ymax)

class Ax(Box):
    def __init__(self, xmin, xmax, ymin, ymax, screen_width, screen_height):
        Box.__init__(self, xmin, xmax, ymin, ymax)
        self.screen_width = screen_width
        self.screen_height = screen_height
    def get_xlim(self):
        return (self.xmin, self.xmax)
    def get_ylim(self):
        return (self.ymin, self.ymax)
    def gca(self):
        return self

    def get_pixel_pos(self, x=None, y=None):
        """return the plot relative pixel position of x and/or y"""
        if x is not None:
            x0 = (x - self.xmin) / (self.xmax - self.xmin) * self.screen_width

        if y is not None:
            y0 = (y - self.ymin) / (self.ymax - self.ymin) * self.screen_height

        if x is not None and y is not None:
            return np.array([x0, y0])
        elif x is not None:
            return x0
        elif y is not None:
            return y0


class BBox(Box):
    """Define a box with methods for checking collisions etc.
    Extents can be returned in data (default) or pixels"""
    def __init__(self, xmin, xmax, ymin, ymax, ax=None):
        Box.__init__(self, xmin, xmax, ymin, ymax)
        self.ax = ax

    def transData(self, ax):
        #print('transdata')
        limits = []
        for lim in self.xmin, self.xmax:
            limits.append(
                (lim-ax.xmin)/ax.width
            )
        for lim in self.ymin, self.ymax:
            limits.append(
                (lim - ax.ymin) / ax.height
            )
        return BBox(*limits)

    # get midpoint, get points inside
    def get_midpoint(bbox):
        cx = (bbox.xmin + bbox.xmax) / 2
        cy = (bbox.ymin + bbox.ymax) / 2
        return cx, cy

    def get_points_inside(bbox, x, y):
        """Return the indices of points inside the given bbox."""
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        x = np.array(x)
        y = np.array(y)
        x_in = np.logical_and(x >= x1, x <= x2)
        y_in = np.logical_and(y >= y1, y <= y2)
        # nonzero gets the index of non-zero elements; indicies where both
        # x_in[i] and y_in[i] are true. Returns a tuple for some reason, hence [0]
        return_value = np.asarray(np.nonzero(x_in & y_in)[0])
        return return_value

    def overlap_bbox_and_point(bbox, xp, yp):
        """Given a bbox that contains a given point, return the (x, y) displacement
        necessary to make the bbox not overlap the point."""
        cx, cy = bbox.get_midpoint()

        dir_x = np.sign(cx - xp)
        dir_y = np.sign(cy - yp)

        if dir_x == -1:
            dx = xp - bbox.xmax
        elif dir_x == 1:
            dx = xp - bbox.xmin
        else:
            dx = 0

        if dir_y == -1:
            dy = yp - bbox.ymax
        elif dir_y == 1:
            dy = yp - bbox.ymin
        else:
            dy = 0
        return dx, dy

    @property
    def extents(self):
        """Return x0, y0, x1, y1 in default data values"""
        # matplotlib returns the x0 etc numbers by their internal order,
        # I don't think we care about that here so I'm just using xmin etc
        # x0, y0, x1, y1
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    @property
    def pixel_extents(self):
        """Return x0, y0, x1, y1 in pixel values"""
        ax = self.ax

        x1, y1 = ax.get_pixel_pos(self.xmin, self.ymin)
        x2, y2 = ax.get_pixel_pos(self.xmax, self.ymax)

        return np.array([x1,y1, x2, y2])


    @staticmethod # why is this a static method in matplotlib?
    def intersection(bbox1, bbox2):
        """
        Return the intersection of the two bboxes or None
        if they do not intersect.
        """
        x0 = np.maximum(bbox1.xmin, bbox2.xmin)
        x1 = np.minimum(bbox1.xmax, bbox2.xmax)
        y0 = np.maximum(bbox1.ymin, bbox2.ymin)
        y1 = np.minimum(bbox1.ymax, bbox2.ymax)
        return BBox(x0, x1, y0, y1) if x0 <= x1 and y0 <= y1 else None

    @property
    def size(self):
        return self.width, self.height

class Text:
    """contains x, y, text and font_sz of labels. Will also act
    as the Box class, with box width defined by:
        len(text)*font_sz*0.8
    assuming font is courier.
        """
    def __init__(self, x, y, text, font_wi, font_hi, indx, va='bottom', ha='left', ax=None):

        self.x, self.y, self.text, self.font_hi, self.font_wi = x, y, text, font_hi, font_wi
        self.ha, self.va = ha, va
        self.index = indx
        self.ax = ax
        self.anchor = None
        self.pixel_anchor = None
        assert va in ('bottom', 'top', 'center')
        assert ha in ('left', 'right', 'center')

    def get_window_extent(self, hexpand = 1.1, vexpand = 1.1, ax=None, transData=False):
        """return BBox with x and y extents taking alignment into account,
        ."""
        ax = self.ax if ax is None else ax
        if ax is None:
            raise AssertionError('Need ax')
        va, ha = self.va, self.ha
        x, y  = self.x, self.y
        fsz = self.font_hi

        # Calculate in pixels and then convert back to data
        x = (x - ax.xmin) / (ax.xmax - ax.xmin) * ax.screen_width
        y = (y - ax.ymin) / (ax.ymax - ax.ymin) * ax.screen_height

        if va == 'bottom':
            ymin = y
            ymax = y + fsz * hexpand
        elif va == 'center':
            ymin = y - fsz * hexpand/2
            ymax = y + fsz * hexpand/2
        else: # va == 'top':
            ymin = y - fsz * hexpand
            ymax = y


        width = len(self.text) * self.font_wi
        if ha == 'left':
            xmin = x
            xmax = x + width * vexpand
        elif ha == 'right':
            xmin = x - width*vexpand
            xmax = x
        else: # ha == 'center':
            halfvex = (width*vexpand)/2
            xmin = x-halfvex
            xmax = x+halfvex

        # convert back to axis relative units
        xmin = xmin / ax.screen_width * (ax.xmax - ax.xmin) + ax.xmin
        xmax = xmax / ax.screen_width * (ax.xmax - ax.xmin) + ax.xmin
        ymin = ymin / ax.screen_height * (ax.ymax - ax.ymin) + ax.ymin
        ymax = ymax / ax.screen_height * (ax.ymax - ax.ymin) + ax.ymin


        bbox = BBox(xmin, xmax, ymin, ymax, ax)
        if not transData:
            return bbox
        return bbox.transData(ax)

    def get_pixel_extent(self, hexpand = 1.1, vexpand = 1.1, ax=None):
        # get window extent then use the bbox method to convert it pixels,
        # not an ideal arrangement.
        return self.get_window_extent(hexpand, vexpand, ax).pixel_extents

    # maybe make it its own function
    #def set_anchor_pos(self):
    #    pass

    def shift_by_alignment(self):
        """Alter .x, .y to match alignment as bokeh only. ONLY CALL AT THE END.
        get_window_extent() takes alignment into account, use this before
        returning the text position to bokeh. Also sets self.anchor_pos
        & self.anchor_pix, the location of the aligned corner
        Returns self."""
        bb = self.get_window_extent(*(1,1),ax=self.ax)

        # Find the position of the corner that is aligned
        anchors = []
        for xmin, ymin, xmax, ymax in (bb.extents, bb.pixel_extents):
            x_aligner = {'left':xmin, 'right':xmax, 'center':xmin+(xmax-xmin)/2}
            y_aligner = {'bottom':ymin, 'top':ymax, 'center':ymin+(ymax-ymin)/2}
            anchors.append(
                np.array([x_aligner[self.ha], y_aligner[self.va]])
            )

        self.anchor = anchors[0]
        self.pixel_anchor = anchors[1]

        xmin, ymin, xmax, ymax = bb.extents
        self.x = xmin
        self.y = ymin
        #print(xmin, ymin)
        return self


    def get_position(self):
        return np.array([self.x, self.y])
    def set_position(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Text({}, {}, '{}', font_sz={})".format(
            self.x, self.y, self.text, self.font_sz
        )


# def get_text_position(text):
#     x, y = text.get_position()
#     return x, y
#     return (ax.xaxis.convert_units(x),
#             ax.yaxis.convert_units(y))


def get_bboxes(objs, expand, transData=False, ax=None):
    debugprint('get_bboxes()')
    # bbox x/ymins/maxs between 0 and 1
    bboxes = [i.get_window_extent(*expand, ax=ax) for i in objs]
    return bboxes


def get_text_obj_from_dict(data:dict, font_hi:int, font_wi:int, x_key, y_key, txt_key, mask=None, ):
    """Get a list of Text objects from a dictionary of data (e.g. a Bokeh
    DataSource.data).

    Args:
        data (dict): a dictionary including x, y and string values for text
            labels.
        font_hi, font_wi (int): the pixel dimensions of the font used.
        x_key, y_key, txt_key: Dict keys for x and y positions, and string
            values.
        mask (str or array): Bool mask, True values will return a Text obj,
            False values will not return anything;
            so len(returned_texts) == sum(mask)
    """
    if mask is not None:
        if type(mask) == str:
            mask = data[mask]
        assert hasattr(mask, '__next__')
    texts = []
    for row_i, (x, y, txt) in enumerate(zip(data[x_key], data[y_key], data[txt_key])):
        if mask and mask[row_i]:
            texts.append(Text(x,y,txt, font_wi, font_hi, row_i))
    return texts


def move_texts(texts, delta_x, delta_y, bboxes=None, ax=None):
    """Supply deltas in data units, updates text .x,.y"""
    debugprint('move_texts()')
    if bboxes is None:
        bboxes = get_bboxes(texts, (1, 1), ax=ax)#, ax=ax)
    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    for i, (text, dx, dy) in enumerate(zip(texts, delta_x, delta_y)):
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        if x1 + dx < xmin:
            dx = 0
        if x2 + dx > xmax:
            dx = 0
        if y1 + dy < ymin:
            dy = 0
        if y2 + dy > ymax:
            dy = 0

        x, y = text.get_position()
        newx = x + dx
        newy = y + dy
        text.set_position(newx, newy)
    debugprint('move_texts() DONE')
    return texts


def optimally_align_text(x, y, texts, expand=(1., 1.), add_bboxes=[], ax=None,
                         direction='xy'):
    """
    For all text objects find alignment that causes the least overlap with
    points and other texts and apply it
    """

    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    bboxes = get_bboxes(texts, expand, ax=ax)#, ax=ax)
    if 'x' not in direction:
        ha = ['']
    else:
        ha = ['left', 'right', 'center']
    if 'y' not in direction:
        va = ['']
    else:
        va = ['bottom', 'top', 'center']
    alignment = list(product(ha, va))
    #    coords = np.array(zip(x, y))
    for i, text in enumerate(texts):
        #        tcoords = np.array(text.get_position()).T
        #        nonself_coords = coords[~np.all(coords==tcoords, axis=1)]
        #        nonself_x, nonself_y = np.split(nonself_coords, 2, axis=1)
        counts = []
        for h, v in alignment:
            if h:
                text.ha = h
            if v:
                text.va = v
            bbox = text.get_window_extent(*expand, ax=ax)
            # get_points_inside returns a single value or nothing, this value is then thrown away.
            c = len(bbox.get_points_inside(x, y))
            intersections = [bbox.intersection(bbox, bbox2) if i != j else None
                             for j, bbox2 in enumerate(bboxes + add_bboxes)]
            intersections = sum([abs(b.width * b.height) if b is not None else 0
                                 for b in intersections])
            # Check for out-of-axes position
            bbox = text.get_window_extent(*expand, ax=ax)
            x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            if x1 < xmin or x2 > xmax or y1 < ymin or y2 > ymax:
                axout = 1
            else:
                axout = 0
            counts.append((axout, c, intersections))
        # Most important: prefer alignments that keep the text inside the axes.
        # If tied, take the alignments that minimize the number of x, y points
        # contained inside the text.
        # Break any remaining ties by minimizing the total area of intersections
        # with all text bboxes and other objects to avoid.
        a, value = min(enumerate(counts), key=itemgetter(1))
        if 'x' in direction:
            text.ha = alignment[a][0]
        if 'y' in direction:
            text.va = alignment[a][1]
        # update bboxes for future iterations

        bboxes[i] = text.get_window_extent(*expand,ax=ax)
    return texts


def repel_text(texts, ax=None, expand=(1.2, 1.2), move=False):
    """
    Repel texts from each other while expanding their bounding boxes by expand
    (x, y), e.g. (1.2, 1.2) would multiply width and height by 1.2.
    """
    debugprint('repel_text()', 1)
    bboxes = get_bboxes(texts, expand, ax=ax)
    xmins = [bbox.xmin for bbox in bboxes]
    xmaxs = [bbox.xmax for bbox in bboxes]
    ymaxs = [bbox.ymax for bbox in bboxes]
    ymins = [bbox.ymin for bbox in bboxes]

    overlaps_x = np.zeros((len(bboxes), len(bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)
    debugprint('repel_text()', 2)
    for i, bbox1 in enumerate(bboxes):
        overlaps = bbox1.get_points_inside(xmins * 2 + xmaxs * 2, (ymins + ymaxs) * 2) % len(bboxes)
        debugprint('repel_text()', 2, 1)
        overlaps = np.unique(overlaps)
        for j in overlaps:
            bbox2 = bboxes[j]
            x, y = bbox1.intersection(bbox1, bbox2).size
            debugprint('repel_text()', 2, 2)
            overlaps_x[i, j] = x
            overlaps_y[i, j] = y
            direction = np.sign(bbox1.extents - bbox2.extents)[:2]
            overlap_directions_x[i, j] = direction[0]
            overlap_directions_y[i, j] = direction[1]
    debugprint('repel_text()', 3)
    move_x = overlaps_x * overlap_directions_x
    move_y = overlaps_y * overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(overlaps_x), np.sum(overlaps_y)
    if move:
        move_texts(texts, delta_x, delta_y, ax=ax)
    return delta_x, delta_y, q



def repel_text_from_bboxes(add_bboxes, texts, ax=None,
                           expand=(1.2, 1.2),
                           move=False):
    """
    Repel texts from other objects bboxes while expanding their (texts)
    bounding boxes by expand (x, y), e.g. (1.2, 1.2) would multiply width and
    height by 1.2.
    Requires a renderer to get the actual sizes of the text, and to that end
    either one needs to be directly provided, or the axes have to be specified,
    and the renderer is then got from the axes object.
    """


    bboxes = get_bboxes(texts, expand, ax=ax)

    overlaps_x = np.zeros((len(bboxes), len(add_bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)

    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(add_bboxes):
            try:
                x, y = bbox1.intersection(bbox1, bbox2).size
                direction = np.sign(bbox1.extents - bbox2.extents)[:2]
                overlaps_x[i, j] = x
                overlaps_y[i, j] = y
                overlap_directions_x[i, j] = direction[0]
                overlap_directions_y[i, j] = direction[1]
            except AttributeError:
                pass

    move_x = overlaps_x * overlap_directions_x
    move_y = overlaps_y * overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(overlaps_x), np.sum(overlaps_y)
    if move:
        move_texts(texts, delta_x, delta_y, bboxes=bboxes, ax=ax)
    return delta_x, delta_y, q


def repel_text_from_points(x, y, texts, ax=None,
                           expand=(1.2, 1.2), move=False):
    """
    Repel texts from all points specified by x and y while expanding their
    (texts'!) bounding boxes by expandby  (x, y), e.g. (1.2, 1.2)
    would multiply both width and height by 1.2.
    Requires a renderer to get the actual sizes of the text, and to that end
    either one needs to be directly provided, or the axes have to be specified,
    and the renderer is then got from the axes object.
    """
    assert len(x) == len(y)

    bboxes = get_bboxes(texts, expand, ax=ax)

    # move_x[i,j] is the x displacement of the i'th text caused by the j'th point
    move_x = np.zeros((len(bboxes), len(x)))
    move_y = np.zeros((len(bboxes), len(x)))
    for i, bbox in enumerate(bboxes):
        xy_in = bbox.get_points_inside(x, y)
        for j in xy_in:
            xp, yp = x[j], y[j]
            dx, dy = bbox.overlap_bbox_and_point(xp, yp)

            move_x[i, j] = dx
            move_y[i, j] = dy

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)
    q = np.sum(np.abs(move_x)), np.sum(np.abs(move_y))
    if move:
        move_texts(texts, delta_x, delta_y, bboxes=bboxes, ax=ax)
    return delta_x, delta_y, q


def repel_text_from_axes(texts, ax=None, bboxes=None, expand=None):

    if expand is None:
        expand = (1, 1)
    if bboxes is None:
        bboxes = get_bboxes(texts, expand=expand, ax=ax)
    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        dx, dy = 0, 0
        if x1 < xmin:
            dx = xmin - x1
        if x2 > xmax:
            dx = xmax - x2
        if y1 < ymin:
            dy = ymin - y1
        if y2 > ymax:
            dy = ymax - y2
        if dx or dy:
            x, y = texts[i].x, texts[i].y
            newx, newy = x + dx, y + dy
            texts[i].set_position(newx, newy)
    return texts


def float_to_tuple(a):
    try:
        a = float(a)
        return (a, a)
    except TypeError:
        assert len(a) == 2
        try:
            b = float(a[0]), float(a[1])
        except TypeError:
            raise TypeError('Force values must be castable to floats')
        return b


def adjust_text(texts, data_lims, chart_size, x=None, y=None,
                add_objects=None,
                expand_text=(1.05, 1.2), expand_points=(1.05, 1.2),
                expand_objects=(1.05, 1.2), expand_align=(1.05, 1.2),
                autoalign='xy', va='center', ha='center',
                force_text=(0.1, 0.25), force_points=(0.2, 0.5),
                force_objects=(0.1, 0.25),
                lim=500, precision=0.01,
                only_move={'points': 'xy', 'text': 'xy', 'objects': 'xy'},
                text_from_text=True, text_from_points=True):
    """Iteratively adjusts the locations of texts.
    Call adjust_text the very last, after all plotting (especially
    anything that can change the axes limits) has been done. This is
    because to move texts the function needs to use the dimensions of
    the axes, and without knowing the final size of the plots the
    results will be completely nonsensical, or suboptimal.
    First moves all texts that are outside the axes limits
    inside. Then in each iteration moves all texts away from each
    other and from points. In the end hides texts and substitutes them
    with annotations to link them to the respective points.

    Return a list of text with updated xy positions

    Args:
        texts (list): a list of text.Text objects to adjust.
        data_lims ( tuple(xmin, xmax, ymin, ymax) ): The axis limits in data
            units. Required.
        chart_size ( tuple(width, height) ): size of the chart area in pixels.
            Required.
        x (seq): x-coordinates of points to repel from; if not provided only
            uses text coordinates
        y (seq): y-coordinates of points to repel from; if not provided only
            uses text coordinates
        add_objects (list): a list of additional matplotlib objects to avoid;
            they must have a .get_window_extent() method
        ax (obj): axes object with the plot; if not provided is determined by
            plt.gca()
        expand_text (seq): a tuple/list/... with 2 multipliers (x, y) by which
            to expand the bounding box of texts when repelling them from each other;
            default (1.05, 1.2)
        expand_points (seq): a tuple/list/... with 2 multipliers (x, y) by which
            to expand the bounding box of texts when repelling them from points;
            default (1.05, 1.2)
        expand_objects (seq): a tuple/list/... with 2 multipliers (x, y) by which
            to expand the bounding box of texts when repelling them from other
            objects; default (1.05, 1.2)
        expand_align (seq): a tuple/list/... with 2 multipliers (x, y) by which
            to expand the bounding box of texts when autoaligning texts;
            default (1.05, 1.2)
        autoalign: If 'xy' or True, the best alignment of all texts will be
            determined in all directions automatically before running the
            iterative adjustment (overriding va and ha); if 'x', will only align
            horizontally, if 'y', vertically; if False, do nothing (i.e.
            preserve va and ha); default 'xy'
        va (str): vertical alignment of texts; default 'center'
        ha (str): horizontal alignment of texts; default 'center'
        force_text (float): the repel force from texts is multiplied by this
            value; default (0.1, 0.25)
        force_points (float): the repel force from points is multiplied by this
            value; default (0.2, 0.5)
        force_objects (float): same as other forces, but for repelling
            additional objects; default (0.1, 0.25)
        lim (int): limit of number of iterations
        precision (float): iterate until the sum of all overlaps along both x
            and y are less than this amount, as a fraction of the total widths
            and heights, respectively. May need to increase for complicated
            situations; default 0.01
        only_move (dict): a dict to restrict movement of texts to only certain
            axes for certain types of overlaps. Valid keys are 'points', 'text',
            and 'objects'. Valid values are '', 'x', 'y', and 'xy'.
            For example, only_move={'points':'y', 'text':'xy', 'objects':'xy'}
            forbids moving texts along the x axis due to overlaps with points.
            Default: everything is allowed.
        text_from_text (bool): whether to repel texts from each other; default
            True
        text_from_points (bool): whether to repel texts from points; default
            True; can be helpful to switch off in extremely crowded plots
    """
    # The actual chart area will be reduced by the axis labels
    ax = Ax(*data_lims, chart_size[0] - 50, chart_size[1] - 25)
    #plt.draw()
    debugprint(1)
    orig_xy = [text.get_position() for text in texts]
    orig_x = [xy[0] for xy in orig_xy]
    orig_y = [xy[1] for xy in orig_xy]
    force_objects = float_to_tuple(force_objects)
    force_text = float_to_tuple(force_text)
    force_points = float_to_tuple(force_points)
    debugprint(2)
    #    xdiff = np.diff(ax.get_xlim())[0]
    #    ydiff = np.diff(ax.get_ylim())[0]

    bboxes = get_bboxes(texts, (1.0, 1.0),ax=ax)
    sum_width = np.sum(list(map(lambda bbox: bbox.width, bboxes)))
    sum_height = np.sum(list(map(lambda bbox: bbox.height, bboxes)))
    if not any(list(map(lambda val: 'x' in val, only_move.values()))):
        precision_x = np.inf
    else:
        precision_x = precision * sum_width
    #
    if not any(list(map(lambda val: 'y' in val, only_move.values()))):
        precision_y = np.inf
    else:
        precision_y = precision * sum_height
    debugprint(3)
    if x is None and y is None:
        x, y = orig_x, orig_y
    elif y is None or x is None:
        raise ValueError('Please specify both x and y, or neither')

    if add_objects is None:
        text_from_objects = False
        add_bboxes = []
    else:
        try:
            add_bboxes = get_bboxes(add_objects, (1, 1), ax=ax)
        except:
            raise ValueError("Can't get bounding boxes from add_objects - is'\
                             it a flat list of matplotlib objects?")
            return
        text_from_objects = True
    for text in texts:
        text.va = va
        text.ha = ha
    # if save_steps:
    #     if add_step_numbers:
    #         plt.title('Before')
    #     plt.savefig('%s%s.%s' % (save_prefix,
    #                              '000a', save_format), format=save_format, dpi=150)
    # elif on_basemap:
    #     ax.draw(r)
    debugprint(4)
    if autoalign:
        if autoalign is True:
            autoalign = 'xy'
        for i in range(2):
            texts = optimally_align_text(x, y, texts, expand=expand_align,
                                         add_bboxes=add_bboxes,
                                         direction=autoalign,
                                         ax=ax)

    # if save_steps:
    #     if add_step_numbers:
    #         plt.title('Autoaligned')
    #     plt.savefig('%s%s.%s' % (save_prefix,
    #                              '000b', save_format), format=save_format, dpi=150)
    # elif on_basemap:
    #     ax.draw(r)
    debugprint(5)
    texts = repel_text_from_axes(texts, ax, expand=expand_points)
    history = [(np.inf, np.inf)] * 10
    # apply each repel in turn
    for i in xrange(lim):
        #        q1, q2 = [np.inf, np.inf], [np.inf, np.inf]
        debugprint(6)
        if text_from_text:
            d_x_text, d_y_text, q1 = repel_text(texts,ax=ax,
                                                expand=expand_text)
        else:
            d_x_text, d_y_text, q1 = [0] * len(texts), [0] * len(texts), (0, 0)
        debugprint(7)
        if text_from_points:
            d_x_points, d_y_points, q2 = repel_text_from_points(x, y, texts,
                                                                ax=ax,
                                                                expand=expand_points)
        else:
            d_x_points, d_y_points, q2 = [0] * len(texts), [0] * len(texts), (0, 0)
        debugprint(8)
        if text_from_objects:
            d_x_objects, d_y_objects, q3 = repel_text_from_bboxes(add_bboxes,
                                                                  texts,
                                                                  ax=ax,
                                                                  expand=expand_objects)
        else:
            d_x_objects, d_y_objects, q3 = [0] * len(texts), [0] * len(texts), (0, 0)
        debugprint(9)
        if only_move:
            if 'text' in only_move:
                if 'x' not in only_move['text']:
                    d_x_text = np.zeros_like(d_x_text)
                if 'y' not in only_move['text']:
                    d_y_text = np.zeros_like(d_y_text)
            if 'points' in only_move:
                if 'x' not in only_move['points']:
                    d_x_points = np.zeros_like(d_x_points)
                if 'y' not in only_move['points']:
                    d_y_points = np.zeros_like(d_y_points)
            if 'objects' in only_move:
                if 'x' not in only_move['objects']:
                    d_x_objects = np.zeros_like(d_x_objects)
                if 'y' not in only_move['objects']:
                    d_y_objects = np.zeros_like(d_y_objects)
        debugprint(10)
        dx = (np.array(d_x_text) * force_text[0] +
              np.array(d_x_points) * force_points[0] +
              np.array(d_x_objects) * force_objects[0])
        dy = (np.array(d_y_text) * force_text[1] +
              np.array(d_y_points) * force_points[1] +
              np.array(d_y_objects) * force_objects[1])
        qx = np.sum([q[0] for q in [q1, q2, q3]])
        qy = np.sum([q[1] for q in [q1, q2, q3]])
        histm = np.max(np.array(history), axis=0)
        history.pop(0)
        history.append((qx, qy))
        texts = move_texts(
            texts, dx, dy,
            bboxes=get_bboxes(texts, (1, 1), ax=ax),
            ax=ax
        )

        # Stop if we've reached the precision threshold, or if the x and y displacement
        # are both greater than the max over the last 10 iterations (suggesting a
        # failure to converge)
        if (qx < precision_x and qy < precision_y) or np.all([qx, qy] >= histm):
            break
        debugprint('iter', i)
    debugprint(11)
    return [t.shift_by_alignment() for t in texts]

