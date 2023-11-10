// Main function called when HTML will be ready
$(document).ready(function(){
    // Connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');

    socket.on('load_finished', function(msg) {
        $('.loader').css('visibility', 'hidden');
        if ($('#hw-data').val() == "None"){
            $('#power-label').css('display', 'none');
            $('#fps-label').css('display', 'none');
        }else{
            $('#power-label').css('display', 'inline-block');
            $('#fps-label').css('display', 'inline-block');
            compute_efficiency()
        }
    });

    socket.on('hw_data', function(msg) {
        pw_stats = '<p>' + msg.pw_stats +'</p>';
        $('#power-label').html(pw_stats);
        fps = '<p> (' + msg.fps +')</p>';
        $('#fps-label').html(fps);
        compute_efficiency()
    });

    $('#detection-mode').on('change', function(e){
        sendRequestMessage(socket, "settings", this.value);
        $('.loader').css('visibility', 'inherit');
        $('#power-label').css('display', 'none');
        $('#fps-label').css('display', 'none');
        $('#loader-text').html("Loading " + this.value + " model...");
    });

    toggle_hw_stats();
});
