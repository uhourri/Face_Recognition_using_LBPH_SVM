$(document).ready(function() {

    $('#buttonid').click(function() {
        $('#fileid').click();
    });

    $("#fileid").change(function(e) {
        var myToast = mdtoast('Processing the image ...', { duration: 4000 });
        console.log('Sending file..')
        dataForm = new FormData()
        dataForm.append('file', $('#fileid')[0].files[0])

        $.ajax({
          type: "POST",
          url: "/upload_file",
          dataType: 'json',
          processData: false,
          contentType: false,
          data: dataForm,
          success: (response) => {
            $("#imgid").attr("src", "data:image/png;base64," + response);
          }
        });
    });
});