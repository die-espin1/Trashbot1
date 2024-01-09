const URL = "https://teachablemachine.withgoogle.com/models/HY3zi1RNm/";

async function createModel() {
    const checkpointURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    const recognizer = speechCommands.create(
        "BROWSER_FFT",
        undefined,
        checkpointURL,
        metadataURL
    );

    await recognizer.ensureModelLoaded();

    return recognizer;
}

async function init() {
    const recognizer = await createModel();
    const classLabels = recognizer.wordLabels();
    const labelContainer = document.getElementById("label-container");
    for (let i = 0; i < classLabels.length; i++) {
        labelContainer.appendChild(document.createElement("div"));
    }

    let currentClass = null;

    recognizer.listen(result => {
        const scores = result.scores;
        for (let i = 0; i < classLabels.length; i++) {
            const classPrediction = classLabels[i] + ": " + scores[i].toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;
        }

        // Check if the class is 0, 1, or 2 and log to the console only when it changes
        if (result.scores[0] > 0.99) {
            if (currentClass !== "Enviria") {
                console.log("Clase cambiada a Enviria");
                currentClass = "Enviria";
                startRecording();  // Inicia la grabación cuando se detecta la clase 0
            }
        } else if (result.scores[1] > 0.95) {
            if (currentClass !== "Clase 1") {
                console.log("Clase cambiada a Clase 1");
                currentClass = "Clase 1";
                // Puedes agregar lógica adicional para la clase 1 aquí si es necesario
            }
        } else if (result.scores[2] > 0.99) {
            if (currentClass !== "Volnuratus non victus") {
                console.log("Clase cambiada a Volnuratus non victus");
                currentClass = "Volnuratus non victus";
                startRecording();  // Inicia la grabación cuando se detecta la clase 2
            }
        }
    }, {
        includeSpectrogram: true,
        probabilityThreshold: 0.95,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.50
    });
}

let blobs = [];
let stream;
let rec;
let recordUrl;
let audioResponseHandler;

function recorder(url, handler) {
    recordUrl = url;
    if (typeof handler !== "undefined") {
        audioResponseHandler = handler;
    }
}

async function record() {
    try {
        document.getElementById("text").innerHTML = "<i>Grabando...</i>";
        document.getElementById("record").style.display="none";
        document.getElementById("stop").style.display="";
        document.getElementById("record-stop-label").style.display="block"
        document.getElementById("record-stop-loading").style.display="none"
        document.getElementById("stop").disabled=false

        blobs = [];

        stream = await navigator.mediaDevices.getUserMedia({audio:true, video:false})
        rec = new MediaRecorder(stream);
        rec.ondataavailable = e => {
            if (e.data) {
                blobs.push(e.data);
            }
        }
        
        rec.onstop = doPreview;
        
        rec.start();
    } catch (e) {
        alert("No fue posible iniciar el grabador de audio. Favor de verificar que se tenga el permiso adecuado, estar en HTTPS, etc...");
    }
}

function doPreview() {
    if (!blobs.length) {
        console.log("No hay blobios!");
    } else {
        console.log("Tenemos blobios!");
        const blob = new Blob(blobs);

        var fd = new FormData();
        fd.append("audio", blob, "audio");

        fetch(recordUrl, {
            method: "POST",
            body: fd,
        })
        .then((response) => response.json())
        .then(audioResponseHandler)
        .catch(err => {
            console.log("Oops: Ocurrió un error", err);
        });
    }
}

function stop() {
    document.getElementById("record-stop-label").style.display="none";
    document.getElementById("record-stop-loading").style.display="block";
    document.getElementById("stop").disabled=true;
    
    rec.stop();
}

function handleAudioResponse(response){
    if (!response || response == null) {
        console.log("No response");
        return;
    }

    document.getElementById("record").style.display="";
    document.getElementById("stop").style.display="none";
    
    if (audioResponseHandler != null) {
        audioResponseHandler(response);
    }
}

function startRecording() {
    // Lógica para iniciar la grabación
    record();
}

init();  // Inicia el reconocimiento de voz al cargar la página
