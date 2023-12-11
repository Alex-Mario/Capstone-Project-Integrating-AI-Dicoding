document.addEventListener('DOMContentLoaded', function() {
    var fileInput = document.getElementById('pdf_file');
    var fileLabel = document.querySelector('.file-upload-label');
    var uploadWarning = document.getElementById('upload-warning'); // Tambahkan ini

    fileInput.addEventListener('change', function() {
        var fileName = this.value.split('\\').pop();
        if (fileName) {
            fileLabel.innerText = 'File selected: ' + fileName;
            // Sembunyikan pesan peringatan saat file sudah dipilih
            uploadWarning.style.display = 'none';
        } else {
            fileLabel.innerText = 'Drag and drop file here or Browse files';
        }
    });
});

function onSubmit() {
    var fileInput = document.getElementById('pdf_file');
    var fileName = fileInput.value.split('\\').pop();
    if (!fileName) {
        // Tampilkan pesan peringatan jika file belum dipilih
        var uploadWarning = document.getElementById('upload-warning');
        uploadWarning.style.display = 'block';
        return false; // Mencegah pengiriman form jika file belum dipilih
    }
    
    document.getElementById("loading").style.display = "block";
    document.getElementById("upload-button").style.display = "none";
}

// Function to hide loading indicator after upload is complete
function onUploadComplete() {
    document.getElementById("loading").style.display = "none";
    document.getElementById("upload-button").style.display = "block";
}
