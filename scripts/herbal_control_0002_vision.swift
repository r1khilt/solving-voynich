// Fixed local feature-print baseline for HERBAL-CONTROL-0002.
import Foundation
import Vision

struct InputRow: Decodable {
    let chapter_class: String
    let manuscript: String
    let crop_file: String
    let crop_sha256: String
}

struct InputManifest: Decodable {
    let rows: [InputRow]
}

func run() throws {
    guard CommandLine.arguments.count == 4 else {
        throw NSError(domain: "usage", code: 1, userInfo: [NSLocalizedDescriptionKey: "Expected ROOT MANIFEST OUTPUT"])
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let manifestURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
    let input = try JSONDecoder().decode(InputManifest.self, from: Data(contentsOf: manifestURL))
    guard input.rows.count == 18 else {
        throw NSError(domain: "inventory", code: 2, userInfo: [NSLocalizedDescriptionKey: "Expected eighteen crops"])
    }

    var observations: [VNFeaturePrintObservation] = []
    for row in input.rows {
        let imageURL = root.appendingPathComponent(row.crop_file)
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = VNGenerateImageFeaturePrintRequestRevision2
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])
        guard let observation = request.results?.first else {
            throw NSError(domain: "vision", code: 3, userInfo: [NSLocalizedDescriptionKey: "No feature print for \(row.crop_file)"])
        }
        observations.append(observation)
    }

    var matrix: [[Double]] = []
    for left in observations {
        var distances: [Double] = []
        for right in observations {
            var distance: Float = 0
            try left.computeDistance(&distance, to: right)
            guard distance.isFinite else {
                throw NSError(domain: "vision", code: 4, userInfo: [NSLocalizedDescriptionKey: "Non-finite feature distance"])
            }
            distances.append(Double(distance))
        }
        matrix.append(distances)
    }
    let result: [String: Any] = [
        "id": "HERBAL-CONTROL-0002",
        "feature_method": "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit",
        "operating_system": ProcessInfo.processInfo.operatingSystemVersionString,
        "row_order": input.rows.map { ["chapter_class": $0.chapter_class, "manuscript": $0.manuscript,
                                        "crop_file": $0.crop_file, "crop_sha256": $0.crop_sha256] },
        "distance_matrix": matrix,
    ]
    let encoded = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    try encoded.write(to: outputURL)
    try "\n".data(using: .utf8)!.appendToFile(outputURL)
    print("feature prints: \(observations.count); output: \(outputURL.path)")
}

extension Data {
    func appendToFile(_ url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: self)
        try handle.close()
    }
}

do {
    try run()
} catch {
    fputs("HERBAL-CONTROL-0002 error: \(error)\n", stderr)
    exit(1)
}
