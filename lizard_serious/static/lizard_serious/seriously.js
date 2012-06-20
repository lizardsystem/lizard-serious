// Javascript

$(document).ready(function() {
    $("a.item-switcher").click(function (event) {
        event.preventDefault();
        //$(".switch-item").css("display", "none");
        //$($(event.currentTarget).data("switch")).css("display", "block");
        $(".switch-item").hide("slow");
        $($(event.currentTarget).data("switch")).show("slow");
    });
});
