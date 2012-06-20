# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.txt.
from lizard_ui.views import UiView
from lizard_map.views import MapView


class SeriousView(UiView):
    template_name = 'lizard_serious/serious.html'


class TimeseriesView(MapView):
    template_name = 'lizard_serious/timeseries.html'

