// Main JS to interact with UI
// Configure Toastr
toastr.options = {
  "closeButton": true,
  "positionClass": "toast-top-full-width",
}

/*
 * Function to send message with data to server side.
 * Sends a reset request to server code and reset JS variables and HTML.
 * @param {io.socket} arg The communication socket
 * @param {string} arg The message type
 * @param {string} arg The message data
*/
function sendRequestMessage(socket, type, data) {
    let msg = {
        'type': type,
        'data' : data };
    socket.emit('request', msg);
}

/*
 * Function to compute efficiency if necessary.
 * It checks if HW data are available, and if 'Efficiency' is selected in
 * combo box.
 * If it is, the method compute efficiency in mJ/frame and set in HTML page.
 * This method could be called safely, it updates HTML only if it's required.
 *
*/
function compute_efficiency() {
    if($('#hw-data').length){
        if ($('#hw-data').val() == "Efficiency"){
            // Extract text from HTML
            let pw_lbl = $('#power-label p').text();
            let fps_lbl = $('#fps-label p').text();

            // Fetch avg power and fps values
            // Regexp will get last int/float values in the string
            let matches = pw_lbl.match(/\d+/g);
            let avg_pw = matches[matches.length-1];
            matches = fps_lbl.match(/\d+.\d+/g);
            let fps = matches[matches.length-1];

            // Compute efficiency
            let eff = parseInt(avg_pw) / parseFloat(fps)
            eff_lbl = '<p>  (Efficiency: ' + eff.toFixed(2) +'mJ/frame)</p>';
            $('#fps-label').html(eff_lbl);
        }
    }
}

/*
 * Function to handle state of hw stats combo box.
 * It checks if HW data are available, it hides/display hw stats.
 * This method could be called safely, it updates HTML only if HW data are
 * available.
*/
function toggle_hw_stats() {
    if($('#hw-data').length){
        if ($('#hw-data').val() == "None"){
            $('#power-label').css('display', 'none');
            $('#fps-label').css('display', 'none');
        }else{
            $('#power-label').css('display', 'inline-block');
            $('#fps-label').css('display', 'inline-block');
            compute_efficiency()
        }

        $('#hw-data').on('change', function(e){
            if (this.value == "None"){
                $('#power-label').css('display', 'none');
                $('#fps-label').css('display', 'none');
            }else{
                $('#power-label').css('display', 'inline-block');
                $('#fps-label').css('display', 'inline-block');
                compute_efficiency()
            }
        });
    }
}
