//
//  ViewController.swift
//  ObjDetector
//
//  Created by Genry on 09.04.2025.
//

import AVFoundation
import CoreImage
import Vision
import UIKit
import TensorFlowLite

struct Detection {
    let confidence: Float
    var bbox: CGRect
}

class ViewController: UIViewController {

    // MARK: - Constants
        
    private struct Constants {
        static let confidence: Float = 0.9
        static let dimenstions: Int = 8400
        static let ultralyticsInputImageSize: CGSize = CGSize(width: 640, height: 640)
        static let pytorchInputImageSize: CGSize = CGSize(width: 1512, height: 2016)
        static let createMlInputImageSize: CGSize = CGSize(width: 832, height: 832)
        static let tfliteInputImageSize: CGSize = CGSize(width: 320, height: 320)
    }
    
    let queue = DispatchQueue(label: "camera.frame.processing", qos: .userInteractive)
    lazy var session: AVCaptureSession = {
       let object = AVCaptureSession()
        object.sessionPreset = .photo
        return object
    }()
    lazy var config: MLModelConfiguration = {
        let object = MLModelConfiguration()
        object.computeUnits = .all
        return object
    }()
    
    let videoOutput = AVCaptureVideoDataOutput()
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    // Overlay layer to hold all bounding box views
    let detectionOverlay = CALayer()
    
    var ratio: CGFloat = 0
    var yOffset: CGFloat = 0
    
    lazy var viewSize: CGSize = { CGSize(width: view.bounds.width, height: view.bounds.width) }()
    let videoView: UIView = {
        let view = UIView()
        view.clipsToBounds = true
        return view
    }()
    
    lazy var createMl: TVRemoteCreateML? = {
        try? TVRemoteCreateML(configuration: config)
    }()
    lazy var ultralytics: TVRemoteUltralytics? = {
        try? TVRemoteUltralytics(configuration: config)
    }()
    
    lazy var pytorch: TVRemoteDetectionPytorch? = {
        try? TVRemoteDetectionPytorch(configuration: config)
    }()
    
    var tflite: Interpreter? = {
        guard let modelPath = Bundle.main.path(forResource: "TVRemoteDetectionTFlite", ofType: "tflite") else {
            return nil
        }
        return try? Interpreter(modelPath: modelPath)
    }()
    
    
    let semaphore = DispatchSemaphore(value: 1)
    var frameCount = 0
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setup()
    }
 
    func setup() {
        // 1. Choose a camera
        guard
            let camera = AVCaptureDevice.default(for: .video),
            let input = try? AVCaptureDeviceInput(device: camera)
        else {
            return
        }
        
        // 2. Add input to session
        session.addInput(input)
        
        // Get dimensions of preview
        let dimensions = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
        ratio = CGFloat(dimensions.width) / CGFloat(dimensions.height)    // 4032 / 3024 == 1,3333333333

        // 3. Setup output for video data (for processing)
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(videoOutput)
        
        // 4. View Container
        let viewHeight = viewSize.width
        yOffset = view.center.y - viewHeight / 2
        let viewRect = CGRect(x: 0, y: yOffset, width: viewSize.width, height: viewHeight)
        videoView.frame = viewRect
        view.addSubview(videoView)
        
        // 5. Create and add preview layer - VIDEO from Camera
        let layerHeight = viewSize.width * ratio
        let layerRect = CGRect(x: 0, y: 0, width: viewSize.width, height: layerHeight)
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = layerRect
        videoView.layer.addSublayer(previewLayer)
        
        // 7. Layer for drawing
        detectionOverlay.frame = previewLayer.bounds
        previewLayer.addSublayer(detectionOverlay)

        // 6. Start running the session
        queue.async { [weak self] in
            guard let self = self else { return }
            self.session.startRunning()
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
       didOutput sampleBuffer: CMSampleBuffer,
       from connection: AVCaptureConnection
    ) {
        frameCount += 1
        guard frameCount % 2 == 0 else { return }
        frameCount = 0
        
        guard semaphore.wait(timeout: .now()) == .success else { return }
 
        guard
            let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
            let resizedBuffer = pixelBuffer.resizePixelBuffer(
                size: Constants.createMlInputImageSize,
                orientation: connection.orientation
            ),
            let model = self.createMl?.model,
            let coreMlModel = try? VNCoreMLModel(for: model)
        else {
            semaphore.signal()
            return
        }
        
        let request = VNCoreMLRequest(model: coreMlModel) { [weak self] request, error in
            DispatchQueue.main.async { [weak self] in
                self?.detectionOverlay.sublayers?.forEach { $0.removeFromSuperlayer() } // clear old boxes
            }
            
            // CreateMl
            if let results = request.results as? [VNRecognizedObjectObservation],
               let detection = self?.processRecognizedObjects(results) {
                DispatchQueue.main.async { [weak self] in
                    self?.drawBoundingBox(detection.bbox, label: "controller", confidence: detection.confidence)
                }
            }
            
//            // Ultralytics
//            if let results = request.results as? [VNCoreMLFeatureValueObservation],
//                let multiArray = results.first?.featureValue.multiArrayValue {
//                
//                if let detection = self?.processUltralytics(multiArray) {
//                    DispatchQueue.main.async { [weak self] in
//                        self?.drawBoundingBox(detection.bbox, label: "controller", confidence: detection.confidence)
//                    }
//                }
//            }
            
            DispatchQueue.main.async { [weak self] in
                self?.semaphore.signal()
            }
        }
        
//        do {
            // Pytorch
//            if let inputArray = processPytorch(pixelBuffer: resizedBuffer),
//               let model = self.pytorch {
//                let modelInput = TVRemoteDetectionPytorchInput(input: inputArray)
//                let prediction = try model.prediction(input: modelInput)
//                
//                self.handlePytorchUI(prediction: prediction)
//            } else {
//                semaphore.signal()
//            }
            
//            //TFlite
//            if let interpreter = tflite {
//                try interpreter.allocateTensors()
//               
//                // CVPixelBuffer to [1, 320, 320, 3]
//                guard let inputBuffer = self.pixelBufferToRGBTensor(pixelBuffer: resizedBuffer) else {
//                    print("Failed to convert pixel buffer.")
//                    semaphore.signal()
//                    return
//                }
//                try interpreter.copy(inputBuffer, toInputAt: 0)
//                // Inference
//                try interpreter.invoke()
//                
//                let labelsTensor = try interpreter.output(at: 0)  // "labels"
//                let boxesTensor = try interpreter.output(at: 1)   // "boxes"
//                let finalBoxesTensor = try interpreter.output(at: 2)  // "final_boxes"
//                
//                handleTFData(labelsTensor: labelsTensor, boxesTensor: boxesTensor, finalBoxesTensor: finalBoxesTensor)
//            } else {
//                semaphore.signal()
//            }
//        } catch {
//            print("Prediction error:", error)
//            semaphore.signal()
//        }
        
        request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: resizedBuffer, orientation: .up, options: [:])
        try? handler.perform([request])
    }
}

// MARK: - TFLite

private extension ViewController {
    func handleTFData(labelsTensor: Tensor, boxesTensor: Tensor, finalBoxesTensor: Tensor) {
        let confidences = labelsTensor.data.toArray(type: Float32.self)
        
        var bestConfidence: Float32 = 0
        var bestIndex: Int = 0
        
        for (i, confidence) in confidences.enumerated() {
            if confidence > bestConfidence {
                bestConfidence = confidence
                bestIndex = i
            }
        }
        let boxes = finalBoxesTensor.data.toArray(type: Float32.self)
        let box = Array(boxes[bestIndex * 4..<(bestIndex * 4 + 4)])
        
        let scaleX = Float(previewLayer.bounds.width)
        let scaleY = Float(previewLayer.bounds.width)
        
        let cx = box[0] * scaleX
        let cy = box[1] * scaleY
        let w  = box[2] * scaleX
        let h  = box[3] * scaleY

        // to CGRect
        let x = cx - w / 2
        let y = cy - h / 2
        let rect = CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(w), height: CGFloat(h))

        if bestConfidence > 0.5 {
            DispatchQueue.main.async {
                self.drawBoundingBox(rect, label: "TV Remote", confidence: bestConfidence)
                self.semaphore.signal()
            }
        } else {
            self.semaphore.signal()
        }
    }
    
    func pixelBufferToRGBTensor(pixelBuffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        var tensorData = Data(count: width * height * 3 * MemoryLayout<Float32>.size)

        tensorData.withUnsafeMutableBytes { ptr in
            for y in 0..<height {
                let row = baseAddress.advanced(by: y * bytesPerRow)
                for x in 0..<width {
                    let pixel = row.advanced(by: x * 4).assumingMemoryBound(to: UInt8.self)
                    let offset = (y * width + x) * 3
                    ptr.storeBytes(of: Float32(pixel[2]) / 255.0 * 2 - 1, toByteOffset: offset * 4, as: Float32.self) // R
                    ptr.storeBytes(of: Float32(pixel[1]) / 255.0 * 2 - 1, toByteOffset: (offset + 1) * 4, as: Float32.self) // G
                    ptr.storeBytes(of: Float32(pixel[0]) / 255.0 * 2 - 1, toByteOffset: (offset + 2) * 4, as: Float32.self) // B
                }
            }
        }
        return tensorData
    }
}

// MARK: - Pytorch

private extension ViewController {
    func mlIndex(_ i: Int...) -> [NSNumber] {
        return i.map { NSNumber(value: $0) }
    }
    func handlePytorchUI(prediction: TVRemoteDetectionPytorchOutput) {
        let labels = prediction.var_890
        let boxes = prediction.var_940
        let deltas = prediction.deltas
        let anchors = prediction.anchors
        
        let numAnchors = labels.shape[1].intValue
        let threshold: Float = 0.5

        var bestScore: Float = 0
        var bestIndex: Int = -1

        // Найдём лучший якорь (с наибольшей вероятностью класса "пульт")
        for i in 0..<numAnchors {
            let score = labels[mlIndex(0, i, 1)].floatValue // класс "пульт"
            if score > threshold && score > bestScore {
                bestScore = score
                bestIndex = i
            }
        }

        guard bestIndex >= 0 else {
            semaphore.signal()
            return
        } // no remote

        let finalBoxX = boxes[mlIndex(0, bestIndex, 0)].floatValue
        let finalBoxY = boxes[mlIndex(0, bestIndex, 1)].floatValue
        let finalBoxW = boxes[mlIndex(0, bestIndex, 2)].floatValue
        let finalBoxH = boxes[mlIndex(0, bestIndex, 3)].floatValue
        
        // Получаем delta и anchor
        let dx = deltas[mlIndex(0, bestIndex, 0)].floatValue
        let dy = deltas[mlIndex(0, bestIndex, 1)].floatValue
        let dw = deltas[mlIndex(0, bestIndex, 2)].floatValue
        let dh = deltas[mlIndex(0, bestIndex, 3)].floatValue

        let anchorX = anchors[mlIndex(0, bestIndex, 0)].floatValue
        let anchorY = anchors[mlIndex(0, bestIndex, 1)].floatValue
        let anchorW = anchors[mlIndex(0, bestIndex, 2)].floatValue
        let anchorH = anchors[mlIndex(0, bestIndex, 3)].floatValue

        let scaleX = Float(previewLayer.bounds.width / Constants.pytorchInputImageSize.width)
        let scaleY = Float(previewLayer.bounds.width / Constants.pytorchInputImageSize.height)
        
        // Apply deltas to anchors
//        let predCenterX = (anchorX + dx * anchorW) * scaleX
//        let predCenterY = (anchorY + dy * anchorH) * scaleY
//        let predWidth   = (anchorW * exp(dw)) * scaleX
//        let predHeight  = (anchorH * exp(dh)) * scaleY
        
        let predCenterX = finalBoxX * scaleX
        let predCenterY = finalBoxY * scaleY
        let predWidth   = finalBoxW * scaleX
        let predHeight  = finalBoxH * scaleY

        let x = predCenterX - predWidth / 2
        let y = predCenterY - predHeight / 2

        let rect = CGRect(x: CGFloat(x), y: CGFloat(y),
                          width: CGFloat(predWidth), height: CGFloat(predHeight))

        DispatchQueue.main.async {
            self.drawBoundingBox(rect, label: "TV Remote", confidence: bestScore)
            self.semaphore.signal()
        }
    }
    
    func processPytorch(pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        // Исходные размеры модели: [2, 3, 2016, 1512] → [batch, channels, height, width]
        let modelHeight = Int(Constants.pytorchInputImageSize.height)
        let modelWidth = Int(Constants.pytorchInputImageSize.width)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

        guard let mlArray = try? MLMultiArray(shape: [2, 3, modelHeight as NSNumber, modelWidth as NSNumber], dataType: .float32) else {
            return nil
        }

        // Первый элемент (batch 0) будет использоваться
        for y in 0..<min(height, modelHeight) {
            for x in 0..<min(width, modelWidth) {
                let offset = y * bytesPerRow + x * 4
                let r = Float(buffer[offset + 2]) / 255.0
                let g = Float(buffer[offset + 1]) / 255.0
                let b = Float(buffer[offset + 0]) / 255.0

                let normR = r * 2 - 1
                let normG = g * 2 - 1
                let normB = b * 2 - 1

                mlArray[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: normR)
                mlArray[[0, 1, y as NSNumber, x as NSNumber]] = NSNumber(value: normG)
                mlArray[[0, 2, y as NSNumber, x as NSNumber]] = NSNumber(value: normB)
            }
        }

        return mlArray
    }
}

// MARK: - Create ML

private extension ViewController {
    func processRecognizedObjects(_ results: [VNRecognizedObjectObservation]) -> Detection? {
        for object in results {
            guard
                let topLabel = object.labels.first,
                Float(topLabel.confidence) > Constants.confidence
            else {
                continue
            }
            
            let viewWidth = previewLayer.bounds.width
            let viewHeight = previewLayer.bounds.width
            
            let boundingBox = object.boundingBox
            let invertedBox = CGRect(
                x: object.boundingBox.minX,
                y: 1 - boundingBox.minY - boundingBox.height,
                width: boundingBox.width,
                height: boundingBox.height
            )
            
            let normalizedViewRect = CGRect(
                x: invertedBox.minX * viewWidth,
                y: invertedBox.minY * viewHeight,
                width: invertedBox.width * viewWidth,
                height: invertedBox.height * viewHeight
            )
            
            return Detection(confidence: Float(topLabel.confidence), bbox: normalizedViewRect)
        }
        return nil
    }
}

// MARK: - Ultralytics

private extension ViewController {
    func processUltralytics(_ multiArray: MLMultiArray) -> Detection? {
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        var detections: [Detection] = []
        
        for i in 0..<Constants.dimenstions {
            let x = CGFloat(ptr[i])
            let y = CGFloat(ptr[i + Constants.dimenstions])
            let w = CGFloat(ptr[i + Constants.dimenstions * 2])
            let h = CGFloat(ptr[i + Constants.dimenstions * 3])
            let confidence = Float(ptr[i + Constants.dimenstions * 4])
            
            if confidence > Constants.confidence {
                detections.append(Detection(confidence: confidence, bbox: CGRect(x: x, y: y, width: w, height: h)))
            }
        }
        
        guard !detections.isEmpty else { return nil }
        
        var detection = Detection(confidence: 0, bbox: CGRect.zero)
        for detectedBox in detections {
            if detection.confidence < detectedBox.confidence {
                detection = detectedBox
            }
        }
        
        let scaleX = previewLayer.bounds.width / Constants.ultralyticsInputImageSize.width
        let scaleY = previewLayer.bounds.width / Constants.ultralyticsInputImageSize.height
        
        let w = detection.bbox.width * scaleX
        let h = detection.bbox.height * scaleY
        
        let transformedBox = CGRect(
            x: detection.bbox.origin.x * scaleX - (w / 2),
            y: detection.bbox.origin.y * scaleY - (h / 2),
            width: w,
            height: h
        )
        detection.bbox = transformedBox
        
        return detection
    }
}
 
// MARK: - Utils

private extension ViewController {
    func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var floatArray = [Float](repeating: 0, count: count)

        for i in 0..<count {
            floatArray[i] = Float(truncating: multiArray[i])
        }

        return floatArray
    }
    
    func drawBoundingBox(_ rect: CGRect, label: String, confidence: Float) {
        self.detectionOverlay.sublayers?.forEach { $0.removeFromSuperlayer() } // clear old boxes
        
        let boxLayer = CALayer()
        boxLayer.frame = rect
        boxLayer.borderColor = UIColor.red.cgColor
        boxLayer.borderWidth = 2
        boxLayer.cornerRadius = 4
        detectionOverlay.addSublayer(boxLayer)
        
        // Add a text label
        let textLayer = CATextLayer()
        textLayer.string = String(format: "%@ (%.1f%%)", label, confidence * 100)
        textLayer.fontSize = 14
        textLayer.foregroundColor = UIColor.red.cgColor
        textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        textLayer.alignmentMode = .center
        textLayer.cornerRadius = 4
        textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y - 20, width: rect.width, height: 20)
        detectionOverlay.addSublayer(textLayer)
    }
}
