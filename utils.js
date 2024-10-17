function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */



function getChatGPTModelViewMatrix() {
    const transformationMatrix = new Float32Array([
        0.17677669, 0.3061862, 0.4330127, 0,
        -0.19134171, 0.17677669, 0.3535534, 0,
        0.2500000, -0.4330127, 0.3061862, 0,
        0.3, -0.25, 0, 1
    ]);
    
    return getTransposeMatrix(transformationMatrix);
}


function getModelViewMatrix() {
    // Helper functions for creating transformation matrices
    function createTranslationMatrix(x, y, z) {
        return [
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1
        ];
    }

    function createScalingMatrix(sx, sy, sz) {
        return [
            sx, 0,  0,  0,
            0, sy,  0,  0,
            0,  0, sz, 0,
            0,  0,  0,  1
        ];
    }

    function createRotationXMatrix(angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return [
            1,   0,    0, 0,
            0, cos, -sin, 0,
            0, sin,  cos, 0,
            0,   0,    0, 1
        ];
    }

    function createRotationYMatrix(angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return [
            cos, 0, sin, 0,
            0,   1,   0, 0,
           -sin, 0, cos, 0,
            0,   0,   0, 1
        ];
    }

    function createRotationZMatrix(angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return [
            cos, -sin, 0, 0,
            sin,  cos, 0, 0,
            0,     0,  1, 0,
            0,     0,  0, 1
        ];
    }

    function multiplyMatrices(a, b) {
        const result = new Array(16).fill(0);
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                for (let k = 0; k < 4; k++) {
                    result[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
                }
            }
        }
        return result;
    }

    // Define transformations
    const translationMatrix = createTranslationMatrix(0.3, -0.25, 0);
    const scalingMatrix = createScalingMatrix(0.5, 0.5, 1);

    const rotationXMatrix = createRotationXMatrix((30 * Math.PI) / 180);
    const rotationYMatrix = createRotationYMatrix((45 * Math.PI) / 180);
    const rotationZMatrix = createRotationZMatrix((60 * Math.PI) / 180);

    // Combine the transformations
    let modelViewMatrix = scalingMatrix;
    modelViewMatrix = multiplyMatrices(rotationXMatrix, modelViewMatrix);
    modelViewMatrix = multiplyMatrices(rotationYMatrix, modelViewMatrix);
    modelViewMatrix = multiplyMatrices(rotationZMatrix, modelViewMatrix);
    modelViewMatrix = multiplyMatrices(translationMatrix, modelViewMatrix);

    // Return the final model view matrix as a Float32Array
    return new Float32Array(modelViewMatrix);
}


function getPeriodicMovement(startTime) {
    // Transformation parameters for the target position
    const targetTranslation = { x: 0.3, y: -0.25, z: 0 };
    const targetScaling = { x: 0.5, y: 0.5, z: 1 };
    const targetRotation = { x: 30 * Math.PI / 180, y: 45 * Math.PI / 180, z: 60 * Math.PI / 180 }; // radians

    // Initial parameters for no transformation
    const initialTranslation = { x: 0, y: 0, z: 0 };
    const initialScaling = { x: 1, y: 1, z: 1 };
    const initialRotation = { x: 0, y: 0, z: 0 };

    // Get current time and calculate elapsed time
    const currentTime = (Date.now() - startTime) / 1000; // convert to seconds
    const cycleTime = currentTime % 10; // 10-second cycle period

    // Calculate interpolation factor (0 to 1 for first 5 seconds, 1 to 0 for next 5 seconds)
    const progress = cycleTime <= 5 ? cycleTime / 5 : 1 - ((cycleTime - 5) / 5);

    // Interpolate each transformation parameter
    const translation = {
        x: initialTranslation.x + progress * (targetTranslation.x - initialTranslation.x),
        y: initialTranslation.y + progress * (targetTranslation.y - initialTranslation.y),
        z: initialTranslation.z + progress * (targetTranslation.z - initialTranslation.z),
    };

    const scaling = {
        x: initialScaling.x + progress * (targetScaling.x - initialScaling.x),
        y: initialScaling.y + progress * (targetScaling.y - initialScaling.y),
        z: initialScaling.z + progress * (targetScaling.z - initialScaling.z),
    };

    const rotation = {
        x: initialRotation.x + progress * (targetRotation.x - initialRotation.x),
        y: initialRotation.y + progress * (targetRotation.y - initialRotation.y),
        z: initialRotation.z + progress * (targetRotation.z - initialRotation.z),
    };

    // Generate the interpolated transformation matrix
    const translationMatrix = [
        1, 0, 0, translation.x,
        0, 1, 0, translation.y,
        0, 0, 1, translation.z,
        0, 0, 0, 1
    ];

    const scalingMatrix = [
        scaling.x, 0, 0, 0,
        0, scaling.y, 0, 0,
        0, 0, scaling.z, 0,
        0, 0, 0, 1
    ];

    const rotationXMatrix = [
        1, 0, 0, 0,
        0, Math.cos(rotation.x), -Math.sin(rotation.x), 0,
        0, Math.sin(rotation.x), Math.cos(rotation.x), 0,
        0, 0, 0, 1
    ];

    const rotationYMatrix = [
        Math.cos(rotation.y), 0, Math.sin(rotation.y), 0,
        0, 1, 0, 0,
        -Math.sin(rotation.y), 0, Math.cos(rotation.y), 0,
        0, 0, 0, 1
    ];

    const rotationZMatrix = [
        Math.cos(rotation.z), -Math.sin(rotation.z), 0, 0,
        Math.sin(rotation.z), Math.cos(rotation.z), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];

    // Multiply matrices in order: Translation * RotationZ * RotationY * RotationX * Scaling
    const modelViewMatrix = multiplyMatrices(
        multiplyMatrices(
            multiplyMatrices(
                multiplyMatrices(translationMatrix, rotationZMatrix),
                rotationYMatrix),
            rotationXMatrix),
        scalingMatrix
    );

    // Return the final matrix as a Float32Array
    return new Float32Array(modelViewMatrix);
}

// Helper function to multiply two 4x4 matrices
function multiplyMatrices(a, b) {
    const result = new Array(16).fill(0);
    for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
            for (let k = 0; k < 4; k++) {
                result[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
            }
        }
    }
    return result;
}




