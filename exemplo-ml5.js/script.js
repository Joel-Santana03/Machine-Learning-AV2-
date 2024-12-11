// Referências aos elementos HTML
const imageInput = document.getElementById('imageInput');
const image = document.getElementById('image');
const result = document.getElementById('result');

// Carregar o modelo MobileNet
let classifier;
ml5.imageClassifier('MobileNet')
  .then((loadedClassifier) => {
    classifier = loadedClassifier;
    console.log('Modelo MobileNet carregado!');
  })
  .catch((error) => console.error('Erro ao carregar o modelo:', error));

// Manipulação do evento de upload de imagem
imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = () => {
      image.src = reader.result; // Define a imagem no elemento <img>
      image.style.display = 'block'; // Exibe a imagem
      classifyImage(); // Classifica a imagem
    };
    reader.readAsDataURL(file);
  }
});

// Função para classificar a imagem
function classifyImage() {
  if (classifier && image.src) {
    classifier.classify(image)
      .then((results) => {
        // Exibe o resultado
        result.innerHTML = `Resultado: ${results[0].label} (Confiança: ${results[0].confidence.toFixed(2)})`;
      })
      .catch((error) => console.error('Erro ao classificar a imagem:', error));
  } else {
    console.log('O modelo ainda não foi carregado ou a imagem está ausente.');
  }
}
