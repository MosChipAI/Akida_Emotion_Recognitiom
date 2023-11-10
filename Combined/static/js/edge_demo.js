function edgeDemoHandler(socket) {
    // Init variables for Edge Demo
    let max_classes;
    let max_shots;
    let neurons;
    let classes = [];
    let nb_shots = [];

    /*
     * Function to reset model for EdgeDemo.
     * Sends a reset request to server code and reset JS variables and HTML.
     * @param {io.socket} arg The communication socket
     */
    function resetEdgeDemoModel() {
        max_classes = $("#max-classes").val();
        max_shots = $("#max-shots").val();
        neurons = $("#max-neurons").val();
        classes = [];
        nb_shots = [];

        for (let i = 0; i < max_classes; i++) {
            nb_shots.push(max_shots);
        }

        $('.btn-learn').html("Learn class<sup>" +
        "0/" + max_classes + "</sup>");

        let msg = {
            'type': "learning",
            'data': "reset",
            'max_classes': max_classes,
            'max_shots': max_shots,
            'neurons': neurons
        };
        socket.emit('request', msg);

        toastr.info('Akida model re-initialised.');
    }

    resetEdgeDemoModel();

    $(".sidebar-item input").on("change", function (event) {
        resetEdgeDemoModel();
    });

    $('.btn-learn').on('click', function(e) {
        if ($('.btn-learn').text() == 'Reset') {
            // Check if 'Reset' button was triggered
            resetEdgeDemoModel();
        } else {
            // Get current class index and class name from input text
            let new_class_idx = classes.length + 1;
            let cur_class_name = $("#class-name-input").val();

            if (cur_class_name == "reset") {
                // Check if text input contains 'reset'
                resetEdgeDemoModel();
            } else if (cur_class_name == "") {
                toastr.warning('Input text box cannot be empty.');
            } else if (classes.includes(cur_class_name)) {
                // Check if class name was already learnt
                if (nb_shots[classes.indexOf(cur_class_name)] > 0) {
                    nb_shots[classes.indexOf(cur_class_name)]--;
                    sendRequestMessage(socket, "learning", cur_class_name);
                } else {
                    toastr.warning('No shots remaining for class: ' +
                        cur_class_name);
                }
            } else if (new_class_idx <= max_classes) {
                // New class, update class index and class list
                $('.btn-learn sup').text(new_class_idx +
                    "/" +
                    max_classes);
                sendRequestMessage(socket, "learning", cur_class_name);
                classes.push($("#class-name-input").val());
                nb_shots[classes.indexOf(cur_class_name)]--;
            } else {
                toastr.warning('No new classes remaining. Learned class: ' +
                    classes.join(','));
            }

            /* If all classes has been learnt, and no more shots were
             * available. Change button label to 'Reset'.
             */
            if (classes.length == max_classes &&
                Math.max.apply(Math, nb_shots) == 0) {
                $('.btn-learn').text("Reset");
            }
        }
    });
}

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

    edgeDemoHandler(socket);

    toggle_hw_stats();
});
