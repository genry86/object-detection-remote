//
//  CVPixelBuffer+Utility.swift
//  ObjDetector
//
//  Created by Genry on 07.05.2025.
//

import AVFoundation
import CoreImage
import Vision
import UIKit

extension CVPixelBuffer {
    func resizePixelBuffer(size: CGSize, orientation: CGImagePropertyOrientation = .right) -> CVPixelBuffer? {
        let pixelBuffer = self
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
  
        // If Vision rotation isn't used
        ciImage = ciImage.oriented(forExifOrientation: Int32(orientation.rawValue))
        
        let originalWidth = ciImage.extent.width
        let originalHeight = ciImage.extent.height
     
        let scaleX = size.width / originalWidth
        let scaleY = size.height / originalHeight
        let scale = max(scaleX, scaleY)
        
        // let resizedImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        
        let scaledHeight = originalHeight * scale
        let offsetY = scaledHeight - size.height
 
        // 1. Scale image to `size`
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        
        // 2. Translate image on top
        let translatedImage = scaledImage.transformed(by: CGAffineTransform(translationX: 0, y: -offsetY))
        
        // 3. Cropp only `size` part
        let croppedImage = translatedImage.cropped(to: CGRect(origin: .zero, size: size))
 
        var resizedPixelBuffer: CVPixelBuffer?
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            CVPixelBufferGetPixelFormatType(pixelBuffer),
            attributes as CFDictionary,
            &resizedPixelBuffer
        )
        
        guard let outputBuffer = resizedPixelBuffer else {
            return nil
        }
       
        let context = CIContext()
        context.render(croppedImage, to: outputBuffer)
        
        return outputBuffer
    }
}
