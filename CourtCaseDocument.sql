-- for merge 
SELECT CourtCaseDocument.CourtCaseID ,
       CourtCaseDocument.DocumentID ,
       Size, DocumentTypeID
FROM dbo.CourtCaseDocument
JOIN dbo.Document ON Document.DocumentID = CourtCaseDocument.DocumentID