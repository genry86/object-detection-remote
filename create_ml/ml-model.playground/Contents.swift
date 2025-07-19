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
    
    // Train the model
    let classifier = try MLImageClassifier(trainingData: data, parameters: parameters)

    // Check quality
    let trainingMetrics = classifier.trainingMetrics
    let validationMetrics = classifier.validationMetrics

    print("description: \(trainingMetrics.description)")
 
    // Save the model
    try classifier.write(to: modelURL)

    print("✅ Model saved successfully!")

} catch {
    print("❌ Error: \(error)")
}
