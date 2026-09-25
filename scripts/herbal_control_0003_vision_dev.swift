// Fixed local feature-print baseline for the development half of HERBAL-CONTROL-0003.
// This binary refuses partial crops and evaluation rows.
import CryptoKit
import Foundation
import Vision

struct InputRow: Decodable {
    let role: String
    let chapter_class: String
    let manuscript: String
    let crop_file: String
    let crop_sha256: String
}

struct InputManifest: Decodable {
    let id: String
    let status: String
    let panel_manifest_sha256: String
    let rows: [InputRow]
}

struct FreezeManifest: Decodable {
    let id: String
    let status: String
    let panel_manifest_sha256: String
    let sources_manifest_sha256: String
    let development_images_sha256: String
    let evaluation_images_sha256: String
    let source_pages: Int
    let crop_pages: Int
}

func hexDigest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func fail(_ code: Int, _ message: String) -> NSError {
    NSError(domain: "HERBAL-CONTROL-0003", code: code,
            userInfo: [NSLocalizedDescriptionKey: message])
}

func run() throws {
    guard CommandLine.arguments.count == 5 else {
        throw fail(1, "Expected ROOT DEVELOPMENT_IMAGE_MANIFEST FULL_INPUT_FREEZE OUTPUT")
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let manifestURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let freezeURL = URL(fileURLWithPath: CommandLine.arguments[3])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[4])
    let manifestBytes = try Data(contentsOf: manifestURL)
    let input = try JSONDecoder().decode(InputManifest.self, from: manifestBytes)
    let freeze = try JSONDecoder().decode(FreezeManifest.self, from: Data(contentsOf: freezeURL))
    let sourcesBytes = try Data(contentsOf: root.appendingPathComponent(
        "data/manifests/herbal_control_0003_sources.json"))
    let evaluationBytes = try Data(contentsOf: root.appendingPathComponent(
        "data/manifests/herbal_control_0003_evaluation_images.json"))
    guard freeze.id == "HERBAL-CONTROL-0003-input-freeze",
          freeze.status == "complete-pre-score-inputs",
          freeze.panel_manifest_sha256 == "fc2bd19a8eb25d16d8f8387a26b551e42c1f221a048d8f8b07a3f20f02c4b725",
          freeze.sources_manifest_sha256 == hexDigest(sourcesBytes),
          freeze.development_images_sha256 == hexDigest(manifestBytes),
          freeze.evaluation_images_sha256 == hexDigest(evaluationBytes),
          freeze.source_pages == 216, freeze.crop_pages == 216 else {
        throw fail(2, "Complete source/evaluation/development input freeze is missing or changed")
    }
    guard input.id == "HERBAL-CONTROL-0003-development",
          input.status == "complete-pre-score-development-crops",
          input.panel_manifest_sha256 == "fc2bd19a8eb25d16d8f8387a26b551e42c1f221a048d8f8b07a3f20f02c4b725",
          input.rows.count == 108 else {
        throw fail(3, "Development crop manifest is incomplete or not source-frozen")
    }
    guard input.rows.allSatisfy({ $0.role == "development_known" || $0.role == "development_unknown" }) else {
        throw fail(4, "Evaluation or non-development crop row supplied")
    }
    let uniqueRows = Set(input.rows.map { "\($0.role)|\($0.chapter_class)|\($0.manuscript)" })
    guard uniqueRows.count == 108,
          input.rows.filter({ $0.role == "development_known" }).count == 72,
          input.rows.filter({ $0.role == "development_unknown" }).count == 36 else {
        throw fail(5, "Development crop roles are incomplete or duplicated")
    }

    var observations: [VNFeaturePrintObservation] = []
    for row in input.rows {
        let imageURL = root.appendingPathComponent(row.crop_file)
        let bytes = try Data(contentsOf: imageURL)
        guard hexDigest(bytes) == row.crop_sha256 else {
            throw fail(6, "Crop hash mismatch: \(row.crop_file)")
        }
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = VNGenerateImageFeaturePrintRequestRevision2
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])
        guard let observation = request.results?.first else {
            throw fail(7, "No Vision feature print: \(row.crop_file)")
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
                throw fail(8, "Non-finite Vision distance")
            }
            distances.append(Double(distance))
        }
        matrix.append(distances)
    }
    let rowOrder: [[String: String]] = input.rows.map { row in
        ["role": row.role, "chapter_class": row.chapter_class,
         "manuscript": row.manuscript, "crop_file": row.crop_file,
         "crop_sha256": row.crop_sha256]
    }
    let result: [String: Any] = [
        "id": "HERBAL-CONTROL-0003-development",
        "input_manifest_sha256": hexDigest(manifestBytes),
        "full_input_freeze_sha256": hexDigest(try Data(contentsOf: freezeURL)),
        "feature_method": "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit",
        "operating_system": ProcessInfo.processInfo.operatingSystemVersionString,
        "row_order": rowOrder,
        "distance_matrix": matrix,
    ]
    let encoded = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    try encoded.write(to: outputURL)
    let handle = try FileHandle(forWritingTo: outputURL)
    try handle.seekToEnd()
    try handle.write(contentsOf: Data("\n".utf8))
    try handle.close()
    print("development feature prints: \(observations.count); output: \(outputURL.path)")
}

do {
    try run()
} catch {
    fputs("HERBAL-CONTROL-0003 development Vision error: \(error)\n", stderr)
    exit(1)
}
