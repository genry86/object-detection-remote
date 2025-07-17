import Cocoa
import CreateML
import Foundation
import PlaygroundSupport

PlaygroundPage.current.needsIndefiniteExecution = true

// Dataset/Training/
// Dataset/Validation/

// Data
let currentDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let folderURL = currentDirectory.appendingPathComponent("Dataset/Training/")

let datasetURL = URL(fileURLWithPath: "/Users/genry/AI/DevicesClassification/Dataset/Training")
let data = MLImageClassifier.DataSource.labeledDirectories(at: datasetURL)
 
// Model
let modelURL = URL(
    fileURLWithPath: "/Users/genry/AI/DevicesClassification/DevicesClassification9.mlmodel"
)

do {
    var parameters = MLImageClassifier.ModelParameters(
        maxIterations: 10000,
        augmentation: [.flip, .exposure, .rotation, .blur, .crop, .noise]
    )
    
    // Обучаем модель
    let classifier = try MLImageClassifier(trainingData: data, parameters: parameters)

    // Проверим качество
    let trainingMetrics = classifier.trainingMetrics
    let validationMetrics = classifier.validationMetrics

    print("description: \(trainingMetrics.description)")
 
    // Сохраняем модель
    try classifier.write(to: modelURL)

    print("✅ Модель успешно сохранена!")

} catch {
    print("❌ Ошибка: \(error)")
}
