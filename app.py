"""
API REST para Clasificación de Imágenes: Gatos, Perros, Pájaros
Versión compatible - Usando Flask-RESTful en lugar de Flask-RESTX
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import logging
import werkzeug

# =============================================================================
# CONFIGURACIÓN DE LA API
# =============================================================================
class APIConfig:
    MODEL_PATH = "models/modelo_mejorado.h5"
    HOST = "localhost"
    PORT = 5000
    IMAGE_SIZE = (224, 224)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_FILE_SIZE = 16 * 1024 * 1024

# =============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = APIConfig.MAX_FILE_SIZE
CORS(app)
api = Api(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODELO DE IA - CARGA Y CONFIGURACIÓN
# =============================================================================
class ImageClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = ['gatos', 'pajaros', 'perros']
        self.load_model()
    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
            
            logger.info(f"🔄 Cargando modelo desde: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("✅ Modelo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error cargando el modelo: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        try:
            image = image.resize(APIConfig.IMAGE_SIZE)
            img_array = np.array(image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Error preprocesando imagen: {str(e)}")
            raise
    
    def predict(self, image):
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            all_probabilities = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# =============================================================================
# INICIALIZAR CLASIFICADOR
# =============================================================================
try:
    classifier = ImageClassifier(APIConfig.MODEL_PATH)
    logger.info("🎯 Clasificador de imágenes inicializado y listo")
except Exception as e:
    logger.error(f"🚨 Error crítico: No se pudo inicializar el clasificador: {str(e)}")
    classifier = None

# =============================================================================
# UTILIDADES DE VALIDACIÓN
# =============================================================================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in APIConfig.ALLOWED_EXTENSIONS

def validate_image(file):
    try:
        if not file or file.filename == '':
            return False, "No se seleccionó ningún archivo"
        
        if not allowed_file(file.filename):
            return False, f"Tipo de archivo no permitido. Use: {', '.join(APIConfig.ALLOWED_EXTENSIONS)}"
        
        try:
            image = Image.open(io.BytesIO(file.read()))
            file.seek(0)
            image.verify()
            file.seek(0)
            return True, "Imagen válida"
        except Exception:
            return False, "El archivo no es una imagen válida"
            
    except Exception as e:
        return False, f"Error validando imagen: {str(e)}"

# =============================================================================
# PARSER PARA RECIBIR ARCHIVOS
# =============================================================================
parser = reqparse.RequestParser()
parser.add_argument('image', 
                   type=werkzeug.datastructures.FileStorage, 
                   location='files',
                   required=True, 
                   help='Imagen a clasificar (PNG, JPG, JPEG)')

# =============================================================================
# ENDPOINTS DE LA API
# =============================================================================
class HealthCheck(Resource):
    def get(self):
        """
        Verifica que la API esté funcionando correctamente
        """
        try:
            status = {
                'status': 'healthy',
                'message': 'API de clasificación de imágenes funcionando',
                'timestamp': datetime.now().isoformat(),
                'model_loaded': classifier is not None
            }
            logger.info("✅ Health check exitoso")
            return status, 200
        except Exception as e:
            logger.error(f"❌ Health check falló: {str(e)}")
            return {'error': str(e)}, 500

class Predict(Resource):
    def post(self):
        """
        Recibe una imagen y devuelve la clasificación
        """
        try:
            logger.info("📥 Recibiendo solicitud de predicción")
            
            if classifier is None:
                return {'error': 'Modelo no disponible', 'success': False}, 500
            
            # Parse arguments
            args = parser.parse_args()
            file = args['image']
            
            if not file:
                return {'error': 'No se envió ninguna imagen', 'success': False}, 400
            
            # Validar la imagen
            is_valid, validation_message = validate_image(file)
            if not is_valid:
                return {'error': validation_message, 'success': False}, 400
            
            # Procesar la imagen
            try:
                image = Image.open(io.BytesIO(file.read()))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                return {'error': f'Error procesando imagen: {str(e)}', 'success': False}, 400
            
            # Realizar predicción
            prediction_result = classifier.predict(image)
            prediction_result['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"🎯 Predicción exitosa: {prediction_result['predicted_class']} "
                       f"({prediction_result['confidence']:.2%})")
            
            return prediction_result, 200
            
        except Exception as e:
            logger.error(f"❌ Error en endpoint /predict: {str(e)}")
            return {'error': f'Error interno del servidor: {str(e)}', 'success': False}, 500

class ModelInfo(Resource):
    def get(self):
        """
        Devuelve información sobre el modelo cargado
        """
        try:
            if classifier is None:
                return {'error': 'Modelo no disponible'}, 500
            
            model_info = {
                'model_name': 'Clasificador de Imágenes - Gatos, Perros, Pájaros',
                'model_path': APIConfig.MODEL_PATH,
                'classes': classifier.class_names,
                'input_size': APIConfig.IMAGE_SIZE,
                'loaded': True,
                'timestamp': datetime.now().isoformat()
            }
            return model_info, 200
            
        except Exception as e:
            return {'error': str(e)}, 500

# =============================================================================
# REGISTRAR ENDPOINTS
# =============================================================================
api.add_resource(HealthCheck, '/health')
api.add_resource(Predict, '/predict')
api.add_resource(ModelInfo, '/models')

# =============================================================================
# MANEJO DE ERRORES
# =============================================================================
@app.errorhandler(413)
def too_large(e):
    return {'error': f'Archivo demasiado grande. Límite: {APIConfig.MAX_FILE_SIZE/(1024*1024)}MB', 'success': False}, 413

@app.errorhandler(404)
def not_found(e):
    return {'error': 'Endpoint no encontrado', 'success': False}, 404

@app.errorhandler(500)
def internal_error(e):
    return {'error': 'Error interno del servidor', 'success': False}, 500

# =============================================================================
# INICIALIZACIÓN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 INICIANDO API REST DE CLASIFICACIÓN DE IMÁGENES")
    print("=" * 60)
    print(f"📍 URL base: http://{APIConfig.HOST}:{APIConfig.PORT}")
    print(f"🎯 Endpoints disponibles:")
    print(f"   GET  /health  - Verificar estado")
    print(f"   POST /predict - Clasificar imagen")
    print(f"   GET  /models  - Información del modelo")
    print(f"🎯 Modelo cargado: {classifier is not None}")
    print(f"📊 Clases: {classifier.class_names if classifier else 'No disponible'}")
    print("=" * 60)
    
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=True
    )