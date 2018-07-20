-- for merge
SELECT DISTINCT cc.CourtCaseID, ccct.ArticleID
FROM dbo.CourtCase cc
LEFT JOIN dbo.CourtCaseParty ccp ON ccp.CourtCaseID = cc.CourtCaseID
LEFT JOIN dbo.CourtCaseCrimeType ccct ON ccct.CourtCasePartyID = ccp.CourtCasePartyID
--GROUP BY cc.CourtCaseID
--WHERE EXISTS (
--SELECT 1 
--FROM dbo.CourtCasePartyRole ccpr
--WHERE ccpr.CourtCasePartyID = ccp.CourtCasePartyID
--AND ccpr.PartyRoleID = 7
--)
WHERE ccct.ArticleID IS NOT NULL
ORDER BY cc.CourtCaseID



