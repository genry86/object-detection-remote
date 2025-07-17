//
//  Data+Utility.swift
//  ObjDetector
//
//  Created by Genry on 17.07.2025.
//

import Foundation

extension Data {
    func toArray<T>(type: T.Type) -> [T] {
        return self.withUnsafeBytes {
            Array(UnsafeBufferPointer<T>(start: $0.baseAddress!.assumingMemoryBound(to: T.self),
                                         count: self.count / MemoryLayout<T>.stride))
        }
    }
}
