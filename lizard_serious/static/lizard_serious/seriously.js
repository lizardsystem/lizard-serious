// Javascript

$(document).ready(function() {
    $("a.item-switcher").click(function (event) {
        event.preventDefault();
        $(".switch-item").css("display", "none");
        $($(event.currentTarget).data("switch")).css("display", "block");
    });
});
