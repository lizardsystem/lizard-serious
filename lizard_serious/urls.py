# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.txt.
from django.conf.urls.defaults import include
from django.conf.urls.defaults import patterns
from django.conf.urls.defaults import url
from django.contrib import admin

from lizard_ui.urls import debugmode_urlpatterns
from lizard_serious.views import SeriousView

admin.autodiscover()


urlpatterns = patterns(
    '',
    url(r'^$',
        SeriousView.as_view(),
        name='lizard_serious_homepage'),
    url(r'^admin/', include(admin.site.urls)),
    # url(r'^something/',
    #     direct.import.views.some_method,
    #     name="name_it"),
    )
urlpatterns += debugmode_urlpatterns()
