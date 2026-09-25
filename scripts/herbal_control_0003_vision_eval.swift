// Fixed Apple Vision feature print for the evaluation half of HERBAL-CONTROL-0003.
// Refuses to inspect evaluation crops until the complete input freeze and
// an independently audited development selection are supplied unchanged.
import CryptoKit
import Foundation
import Vision

struct EvaluationRow: Decodable {
    let role: String
    let chapter_class: String
    let manuscript: String
    let crop_file: String
    let crop_sha256: String
}

struct EvaluationManifest: Decodable {
    let id: String
    let status: String
    let panel_manifest_sha256: String
    let rows: [EvaluationRow]
}

struct InputFreeze: Decodable {
    let id: String
    let status: String
    let panel_manifest_sha256: String
    let sources_manifest_sha256: String
    let development_images_sha256: String
    let evaluation_images_sha256: String
    let source_pages: Int
    let crop_pages: Int
}

struct DevelopmentSelection: Decodable {
    let id: String
    let status: String
    let panel_manifest_sha256: String
    let full_input_freeze_sha256: String
    let development_features_sha256: String
    let primary_method: String
}

struct DevelopmentAudit: Decodable {
    let id: String
    let status: String
    let score_sha256: String
    let full_input_freeze_sha256: String
    let feature_sha256: String
    let primary_method: String
}

func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func failure(_ code: Int, _ message: String) -> NSError {
    NSError(domain: "HERBAL-CONTROL-0003-evaluation", code: code,
            userInfo: [NSLocalizedDescriptionKey: message])
}

func run() throws {
    guard CommandLine.arguments.count == 7 else {
        throw failure(1, "Expected ROOT EVALUATION_IMAGES INPUT_FREEZE DEVELOPMENT_SELECTION DEVELOPMENT_AUDIT OUTPUT")
    }
    let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let manifestURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let freezeURL = URL(fileURLWithPath: CommandLine.arguments[3])
    let selectionURL = URL(fileURLWithPath: CommandLine.arguments[4])
    let auditURL = URL(fileURLWithPath: CommandLine.arguments[5])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[6])
    let manifestBytes = try Data(contentsOf: manifestURL)
    let freezeBytes = try Data(contentsOf: freezeURL)
    let selectionBytes = try Data(contentsOf: selectionURL)
    let auditBytes = try Data(contentsOf: auditURL)
    let sourceBytes = try Data(contentsOf: root.appendingPathComponent(
        "data/manifests/herbal_control_0003_sources.json"))
    let developmentBytes = try Data(contentsOf: root.appendingPathComponent(
        "data/manifests/herbal_control_0003_development_images.json"))
    let input = try JSONDecoder().decode(EvaluationManifest.self, from: manifestBytes)
    let freeze = try JSONDecoder().decode(InputFreeze.self, from: freezeBytes)
    let selection = try JSONDecoder().decode(DevelopmentSelection.self, from: selectionBytes)
    let audit = try JSONDecoder().decode(DevelopmentAudit.self, from: auditBytes)

    guard freeze.id == "HERBAL-CONTROL-0003-input-freeze",
          freeze.status == "complete-pre-score-inputs",
          freeze.panel_manifest_sha256 == "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832",
          freeze.sources_manifest_sha256 == hash(sourceBytes),
          freeze.development_images_sha256 == hash(developmentBytes),
          freeze.evaluation_images_sha256 == hash(manifestBytes),
          freeze.source_pages == 216, freeze.crop_pages == 216 else {
        throw failure(2, "Complete source/crop freeze is missing or changed")
    }
    guard selection.id == "HERBAL-CONTROL-0003-development-selection",
          selection.status == "development-score-produced-awaiting-audit-and-push",
          selection.panel_manifest_sha256 == freeze.panel_manifest_sha256,
          selection.full_input_freeze_sha256 == hash(freezeBytes),
          audit.id == "HERBAL-CONTROL-0003-independent-development-score-audit",
          audit.status == "pass",
          audit.score_sha256 == hash(selectionBytes),
          audit.full_input_freeze_sha256 == hash(freezeBytes),
          audit.feature_sha256 == selection.development_features_sha256,
          audit.primary_method == selection.primary_method else {
        throw failure(3, "Audited development selection is missing or changed")
    }
    guard input.id == "HERBAL-CONTROL-0003-evaluation",
          input.status == "complete-pre-score-evaluation-crops",
          input.panel_manifest_sha256 == freeze.panel_manifest_sha256,
          input.rows.count == 108 else {
        throw failure(4, "Evaluation crop manifest is incomplete or not source-frozen")
    }
    let validRoles = input.rows.allSatisfy {
        $0.role == "evaluation_known" || $0.role == "evaluation_unknown"
    }
    let uniqueRows = Set(input.rows.map { "\($0.role)|\($0.chapter_class)|\($0.manuscript)" })
    guard validRoles, uniqueRows.count == 108,
          input.rows.filter({ $0.role == "evaluation_known" }).count == 72,
          input.rows.filter({ $0.role == "evaluation_unknown" }).count == 36 else {
        throw failure(5, "Evaluation crop roles are incomplete or duplicated")
    }

    var observations: [VNFeaturePrintObservation] = []
    for row in input.rows {
        let imageURL = root.appendingPathComponent(row.crop_file)
        let bytes = try Data(contentsOf: imageURL)
        guard hash(bytes) == row.crop_sha256 else {
            throw failure(6, "Crop hash mismatch: \(row.crop_file)")
        }
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = VNGenerateImageFeaturePrintRequestRevision2
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])
        guard let feature = request.results?.first else {
            throw failure(7, "No Vision feature print: \(row.crop_file)")
        }
        observations.append(feature)
    }

    var matrix: [[Double]] = []
    for left in observations {
        var distances: [Double] = []
        for right in observations {
            var distance: Float = 0
            try left.computeDistance(&distance, to: right)
            guard distance.isFinite else {
                throw failure(8, "Non-finite Vision distance")
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
        "id": "HERBAL-CONTROL-0003-evaluation",
        "input_manifest_sha256": hash(manifestBytes),
        "full_input_freeze_sha256": hash(freezeBytes),
        "development_selection_sha256": hash(selectionBytes),
        "development_audit_sha256": hash(auditBytes),
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
    print("evaluation feature prints: \(observations.count); output: \(outputURL.path)")
}

do {
    try run()
} catch {
    fputs("HERBAL-CONTROL-0003 evaluation Vision error: \(error)\n", stderr)
    exit(1)
}
