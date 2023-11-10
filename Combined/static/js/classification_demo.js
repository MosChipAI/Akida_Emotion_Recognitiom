// Main function called when HTML will be ready
$(document).ready(function(){
    // Connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');

    // Receive details from server to update akida outcome in HTML
    socket.on('newlabel', function(msg) {
        numbers_string = '<h2>' + msg.number +'</h2>';
        $('#output-label').html(numbers_string);
    });

    socket.on('hw_data', function(msg) {
        pw_stats = '<p>' + msg.pw_stats +'</p>';
        $('#power-label').html(pw_stats);
        fps = '<p> (' + msg.fps +')</p>';
        $('#fps-label').html(fps);
        compute_efficiency()
    });

    toggle_hw_stats();

});
