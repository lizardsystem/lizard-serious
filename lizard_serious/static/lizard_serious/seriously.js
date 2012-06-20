// Javascript
function show_serious_popup() {
    $("#movable-dialog-content").load('/static_media/lizard_serious/popup.html');
    $("#movable-dialog").dialog("open");
    window.setTimeout(reloadGraphs, 150);
}



function serious_popup_click_handler(x, y, map) {
  show_serious_popup();
}

$(document).ready(function() {
    $("a.item-switcher").click(function (event) {
        event.preventDefault();
        //$(".switch-item").css("display", "none");
        //$($(event.currentTarget).data("switch")).css("display", "block");
        $(".switch-item").hide("slow");
        $($(event.currentTarget).data("switch")).show("slow");
    });
});
