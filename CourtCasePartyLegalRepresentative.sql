-- for merge
SELECT CourtCase.CourtCaseID, COUNT(CourtCasePartyLegalRepresentativeID) AS CountOfLegalRepresentative
FROM dbo.CourtCase
LEFT JOIN dbo.CourtCaseParty ON CourtCaseParty.CourtCaseID = CourtCase.CourtCaseID
LEFT JOIN dbo.CourtCasePartyLegalRepresentative ON CourtCasePartyLegalRepresentative.CourtCasePartyID = CourtCaseParty.CourtCasePartyID
WHERE IsActive = 1 AND IsPartyActive = 1
GROUP BY CourtCase.CourtCaseID