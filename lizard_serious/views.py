# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.txt.
from lizard_ui.views import UiView
from lizard_maptree.views import MaptreeHomepageView

HARDCODED_WMS_CATEGORY_SLUG = 'neo'

class SeriousView(UiView):
    template_name = 'lizard_serious/serious.html'


class TimeseriesView(MaptreeHomepageView):
    template_name = 'lizard_serious/timeseries.html'
    root_slug = HARDCODED_WMS_CATEGORY_SLUG
    page_title = 'Serious tijdreeksen'


