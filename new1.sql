USE [DEV-RWA_IECMS-DE_Live_20180703_ddxk]
GO
/****** Object:  StoredProcedure [dbo].[sync_UpdateReportingCourtCase]    Script Date: 7/31/2018 2:34:03 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
 ALTER PROC [dbo].[sync_UpdateReportingCourtCase] AS BEGIN 
 SET NOCOUNT ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCase]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCase] ( notRegisteredCaseCode, 
relatedLaw, 
caseSubmittedDate, 
countOfJudgmentPages, 
CourtCaseID, 
appealedCourtCaseID, 
chiefRegistrarUserId, 
subjectMatter, 
chainId, 
caseInitialId, 
committedByMinor, 
specialCaseId, 
IsPublicCase, 
combineCaseNumber, 
UpdatedUserId, 
genderBasedViolence, 
OwnerUserId, 
LitigationCaseID, 
caseRejectionId, 
executionCaseApprovedUserId, 
held, 
judgeDecision, 
instanceLevelId, 
subCategoryId, 
courtPresidentUserId, 
receiptDocumentId, 
notRegisteredCaseNumber, 
courtId, 
prosecutionCaseID, 
hasPassedCaseNumberAllocated, 
groundOfClaim, 
WFActionId, 
assignedRegistrarUserId, 
solvedFromAbunzi, 
casePriorityId, 
extraOrdinaryProcedureId, 
decisionNumber, 
majorVersion, 
attachedDate, 
colorId, 
rejectionDetails, 
statutoryInstruments, 
DateUpdated, 
categoryId, 
PublicOwnerUserId, 
caseCode, 
WFActionDate, 
value, 
paymentBankId, 
initiatedFromAbunzi, 
decisionDate, 
PreviousCourtCaseID, 
assignedJudgeUserId, 
caseNumber, 
externalCases, 
NonDraftCaseNumberGenerationDate, 
facts, 
WFStateId, 
isDetentionCase, 
caseRegisteredDate, 
isExempted, 
receiptNumber, 
CreatedUserID, 
DateCreated )
 SELECT [DE_CourtCase].notRegisteredCaseCode, 
 [DE_CourtCase].relatedLaw, 
 [DE_CourtCase].caseSubmittedDate, 
 [DE_CourtCase].countOfJudgmentPages, 
 [DE_CourtCase].CourtCaseInstanceId, 
 [DE_CourtCase].appealedCourtCaseInstanceId, 
 [DE_CourtCase].chiefRegistrarUserId, 
 [DE_CourtCase].subjectMatter, 
 [DE_CourtCase].chainId, 
 [DE_CourtCase].caseInitialId, 
 [DE_CourtCase].committedByMinor, 
 [DE_CourtCase].specialCaseId, 
 [DE_CourtCase].IsPublicCase, 
 [DE_CourtCase].combineCaseNumber, 
 [DE_CourtCase].UpdatedUserId, 
 [DE_CourtCase].genderBasedViolence, 
 [DE_CourtCase].OwnerUserId, 
 [DE_CourtCase].LitigationCaseInstanceId, 
 [DE_CourtCase].caseRejectionId, 
 [DE_CourtCase].executionCaseApprovedUserId, 
 [DE_CourtCase].held, 
 [DE_CourtCase].judgeDecision, 
 [DE_CourtCase].instanceLevelId, 
 [DE_CourtCase].subCategoryId, 
 [DE_CourtCase].courtPresidentUserId, 
 [DE_CourtCase].receiptDocumentId, 
 [DE_CourtCase].notRegisteredCaseNumber, 
 [DE_CourtCase].courtId, 
 [DE_CourtCase].prosecutionCaseInstanceId, 
 [DE_CourtCase].hasPassedCaseNumberAllocated, 
 [DE_CourtCase].groundOfClaim, 
 [DE_CourtCase].WFActionId, 
 [DE_CourtCase].assignedRegistrarUserId, 
 [DE_CourtCase].solvedFromAbunzi, 
 [DE_CourtCase].casePriorityId, 
 [DE_CourtCase].extraOrdinaryProcedureId, 
 [DE_CourtCase].decisionNumber, 
 [DE_CourtCase].majorVersion, 
 [DE_CourtCase].attachedDate, 
 [DE_CourtCase].colorId, 
 [DE_CourtCase].rejectionDetails, 
 [DE_CourtCase].statutoryInstruments, 
 [DE_CourtCase].DateUpdated, 
 [DE_CourtCase].categoryId, 
 [DE_CourtCase].PublicOwnerUserId, 
 [DE_CourtCase].caseCode, 
 [DE_CourtCase].WFActionDate, 
 [DE_CourtCase].value, 
 [DE_CourtCase].paymentBankId, 
 [DE_CourtCase].initiatedFromAbunzi, 
 [DE_CourtCase].decisionDate, 
 [DE_CourtCase].PreviousCourtCaseInstanceId, 
 [DE_CourtCase].assignedJudgeUserId, 
 [DE_CourtCase].caseNumber, 
 [DE_CourtCase].externalCases, 
 [DE_CourtCase].NonDraftCaseNumberGenerationDate, 
 [DE_CourtCase].facts, 
 [DE_CourtCase].WFStateId, 
 [DE_CourtCase].isDetentionCase, 
 [DE_CourtCase].caseRegisteredDate, 
 [DE_CourtCase].isExempted, 
 [DE_CourtCase].receiptNumber, [DE_CourtCasePublishedItem].CreatedUserID, 
[DE_CourtCasePublishedItem].DateCreated
 FROM [DE_CourtCase]
 INNER JOIN [DE_CourtCasePublishedItem]
 ON [DE_CourtCasePublishedItem].courtCaseId = [DE_CourtCase].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBefore] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBefore]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBefore] ( courtCaseAppearBeforeId, 
isValidated, 
atCourtAppearDate, 
documentId, 
attachedDate, 
dateUpdated, 
orderNo, 
orderDate, 
registrarUserId, CourtCaseID )
 SELECT [DE_CourtCaseAppearBefore].courtCaseAppearBeforeId, 
[DE_CourtCaseAppearBefore].isValidated, 
[DE_CourtCaseAppearBefore].atCourtAppearDate, 
[DE_CourtCaseAppearBefore].documentId, 
[DE_CourtCaseAppearBefore].attachedDate, 
[DE_CourtCaseAppearBefore].dateUpdated, 
[DE_CourtCaseAppearBefore].orderNo, 
[DE_CourtCaseAppearBefore].orderDate, 
[DE_CourtCaseAppearBefore].registrarUserId, CourtCaseInstanceID
 FROM [DE_CourtCaseAppearBefore] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAppearBefore].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBefore] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBeforeParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBeforeParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBeforeParty] ( partyID, 
partyRoleId, 
courtCaseAppearBeforePartyId, 
courtCaseAppearBeforeId, 
appear )
 SELECT [DE_CourtCaseAppearBeforeParty].partyInstanceId, 
[DE_CourtCaseAppearBeforeParty].partyRoleId, 
[DE_CourtCaseAppearBeforeParty].courtCaseAppearBeforePartyId, 
[DE_CourtCaseAppearBeforeParty].courtCaseAppearBeforeId, 
[DE_CourtCaseAppearBeforeParty].appear
 FROM [DE_CourtCaseAppearBeforeParty] INNER JOIN [DE_CourtCaseAppearBefore] ON [DE_CourtCaseAppearBeforeParty].courtCaseAppearBeforeId = [DE_CourtCaseAppearBefore].courtCaseAppearBeforeId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAppearBefore].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppearBeforeParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummary] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummary]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummary] ( partyID, 
dummyPartyId, 
partyRoleId, 
dateUpdated, 
updatedUserId, 
caseSummary, 
courtCaseSummaryId, 
isValidated, CourtCaseID )
 SELECT [DE_CourtCaseSummary].partyInstanceId, 
[DE_CourtCaseSummary].dummyPartyId, 
[DE_CourtCaseSummary].partyRoleId, 
[DE_CourtCaseSummary].dateUpdated, 
[DE_CourtCaseSummary].updatedUserId, 
[DE_CourtCaseSummary].caseSummary, 
[DE_CourtCaseSummary].courtCaseSummaryId, 
[DE_CourtCaseSummary].isValidated, CourtCaseInstanceID
 FROM [DE_CourtCaseSummary] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSummary].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummary] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseRegistrarReport] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseRegistrarReport]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseRegistrarReport] ( dateOfJudgment, 
dateAttached, 
dateUpdated, 
reportDate, 
isDatesObeyed, 
hearingDate, 
updatedUserId, 
isValidated, 
registrarUserId, 
mediationDocumentId, 
courtCaseRegistrarReportId, 
signedCopyDocumentId, 
description, 
isSuccessfulMediation, CourtCaseID )
 SELECT [DE_CourtCaseRegistrarReport].dateOfJudgment, 
[DE_CourtCaseRegistrarReport].dateAttached, 
[DE_CourtCaseRegistrarReport].dateUpdated, 
[DE_CourtCaseRegistrarReport].reportDate, 
[DE_CourtCaseRegistrarReport].isDatesObeyed, 
[DE_CourtCaseRegistrarReport].hearingDate, 
[DE_CourtCaseRegistrarReport].updatedUserId, 
[DE_CourtCaseRegistrarReport].isValidated, 
[DE_CourtCaseRegistrarReport].registrarUserId, 
[DE_CourtCaseRegistrarReport].mediationDocumentId, 
[DE_CourtCaseRegistrarReport].courtCaseRegistrarReportId, 
[DE_CourtCaseRegistrarReport].signedCopyDocumentId, 
[DE_CourtCaseRegistrarReport].description, 
[DE_CourtCaseRegistrarReport].isSuccessfulMediation, CourtCaseInstanceID
 FROM [DE_CourtCaseRegistrarReport] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseRegistrarReport].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseRegistrarReport] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAllParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAllParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAllParty] ( courtCaseAllPartyId, 
partyID, CourtCaseID )
 SELECT [DE_CourtCaseAllParty].courtCaseAllPartyId, 
[DE_CourtCaseAllParty].partyInstanceId, CourtCaseInstanceID
 FROM [DE_CourtCaseAllParty] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAllParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAllParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseWFAction] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseWFAction]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseWFAction] ( userId, 
details, 
actionId, 
actionDate, 
courtCaseWFActionId, 
resultingStateId, CourtCaseID )
 SELECT [DE_CourtCaseWFAction].userId, 
[DE_CourtCaseWFAction].details, 
[DE_CourtCaseWFAction].actionId, 
[DE_CourtCaseWFAction].actionDate, 
[DE_CourtCaseWFAction].courtCaseWFActionId, 
[DE_CourtCaseWFAction].resultingStateId, CourtCaseInstanceID
 FROM [DE_CourtCaseWFAction] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseWFAction].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseWFAction] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFinalDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFinalDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFinalDocument] ( documentId, 
isActive, 
isNoteOrAttachment, 
courtCaseFinalDocumentId, 
isShareWithPublic, CourtCaseID )
 SELECT [DE_CourtCaseFinalDocument].documentId, 
[DE_CourtCaseFinalDocument].isActive, 
[DE_CourtCaseFinalDocument].isNoteOrAttachment, 
[DE_CourtCaseFinalDocument].courtCaseFinalDocumentId, 
[DE_CourtCaseFinalDocument].isShareWithPublic, CourtCaseInstanceID
 FROM [DE_CourtCaseFinalDocument] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseFinalDocument].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFinalDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombine] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombine]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombine] ( attachedDate, 
hearingOf, 
combiningCaseNo, 
orderNo, 
courtCaseCombineId, 
dateUpdated, 
isValidated, 
datedFrom, 
registrarUserId, 
orderDate, 
documentId, CourtCaseID )
 SELECT [DE_CourtCaseCombine].attachedDate, 
[DE_CourtCaseCombine].hearingOf, 
[DE_CourtCaseCombine].combiningCaseNo, 
[DE_CourtCaseCombine].orderNo, 
[DE_CourtCaseCombine].courtCaseCombineId, 
[DE_CourtCaseCombine].dateUpdated, 
[DE_CourtCaseCombine].isValidated, 
[DE_CourtCaseCombine].datedFrom, 
[DE_CourtCaseCombine].registrarUserId, 
[DE_CourtCaseCombine].orderDate, 
[DE_CourtCaseCombine].documentId, CourtCaseInstanceID
 FROM [DE_CourtCaseCombine] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCombine].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombine] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineCase] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineCase]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineCase] ( courtGroupId, 
caseNumber, 
courtCaseID, 
dateFiled, 
courtCaseCombineId, 
courtCaseCombineCaseId )
 SELECT [DE_CourtCaseCombineCase].courtGroupId, 
[DE_CourtCaseCombineCase].caseNumber, 
[DE_CourtCaseCombineCase].courtCaseInstanceId, 
[DE_CourtCaseCombineCase].dateFiled, 
[DE_CourtCaseCombineCase].courtCaseCombineId, 
[DE_CourtCaseCombineCase].courtCaseCombineCaseId
 FROM [DE_CourtCaseCombineCase] INNER JOIN [DE_CourtCaseCombine] ON [DE_CourtCaseCombineCase].courtCaseCombineId = [DE_CourtCaseCombine].courtCaseCombineId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCombine].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineCase] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineParty] ( partyRoleId, 
isApplicant, 
partyID, 
CourtCaseCombinePartyID, 
courtCaseCombineId )
 SELECT [DE_CourtCaseCombineParty].partyRoleId, 
[DE_CourtCaseCombineParty].isApplicant, 
[DE_CourtCaseCombineParty].partyInstanceId, 
[DE_CourtCaseCombineParty].CourtCaseCombinePartyID, 
[DE_CourtCaseCombineParty].courtCaseCombineId
 FROM [DE_CourtCaseCombineParty] INNER JOIN [DE_CourtCaseCombine] ON [DE_CourtCaseCombineParty].courtCaseCombineId = [DE_CourtCaseCombine].courtCaseCombineId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCombine].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCombineParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSchedule] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSchedule]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSchedule] ( hearingStatusId, 
dateUpdated, 
documentId, 
issueDate, 
registrarUserId, 
attachedDate, 
courtCaseScheduleId, 
comments, 
isValidated, 
hearingTypeId, 
hearingDate, CourtCaseID )
 SELECT [DE_CourtCaseSchedule].hearingStatusId, 
[DE_CourtCaseSchedule].dateUpdated, 
[DE_CourtCaseSchedule].documentId, 
[DE_CourtCaseSchedule].issueDate, 
[DE_CourtCaseSchedule].registrarUserId, 
[DE_CourtCaseSchedule].attachedDate, 
[DE_CourtCaseSchedule].courtCaseScheduleId, 
[DE_CourtCaseSchedule].comments, 
[DE_CourtCaseSchedule].isValidated, 
[DE_CourtCaseSchedule].hearingTypeId, 
[DE_CourtCaseSchedule].hearingDate, CourtCaseInstanceID
 FROM [DE_CourtCaseSchedule] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSchedule].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSchedule] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseScheduleHearingParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseScheduleHearingParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseScheduleHearingParty] ( partyID, 
courtCaseScheduleId, 
courtCaseScheduleHearingPartyId )
 SELECT [DE_CourtCaseScheduleHearingParty].partyInstanceId, 
[DE_CourtCaseScheduleHearingParty].courtCaseScheduleId, 
[DE_CourtCaseScheduleHearingParty].courtCaseScheduleHearingPartyId
 FROM [DE_CourtCaseScheduleHearingParty] INNER JOIN [DE_CourtCaseSchedule] ON [DE_CourtCaseScheduleHearingParty].courtCaseScheduleId = [DE_CourtCaseSchedule].courtCaseScheduleId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSchedule].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseScheduleHearingParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCase] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCase]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCase] ( caseNumber, 
dateFiled, 
courtCaseAppealedCaseId, 
courtGroupId, CourtCaseID )
 SELECT [DE_CourtCaseAppealedCase].caseNumber, 
[DE_CourtCaseAppealedCase].dateFiled, 
[DE_CourtCaseAppealedCase].courtCaseAppealedCaseId, 
[DE_CourtCaseAppealedCase].courtGroupId, CourtCaseInstanceID
 FROM [DE_CourtCaseAppealedCase] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAppealedCase].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCase] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCaseDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCaseDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCaseDocument] ( courtCaseAppealedCaseDocumentId, 
documentId, 
courtCaseAppealedCaseId )
 SELECT [DE_CourtCaseAppealedCaseDocument].courtCaseAppealedCaseDocumentId, 
[DE_CourtCaseAppealedCaseDocument].documentId, 
[DE_CourtCaseAppealedCaseDocument].courtCaseAppealedCaseId
 FROM [DE_CourtCaseAppealedCaseDocument] INNER JOIN [DE_CourtCaseAppealedCase] ON [DE_CourtCaseAppealedCaseDocument].courtCaseAppealedCaseId = [DE_CourtCaseAppealedCase].courtCaseAppealedCaseId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAppealedCase].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAppealedCaseDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReference] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReference]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReference] ( internalCaseID, 
courtCaseReferenceId, CourtCaseID )
 SELECT [DE_CourtCaseReference].internalCaseInstanceId, 
[DE_CourtCaseReference].courtCaseReferenceId, CourtCaseInstanceID
 FROM [DE_CourtCaseReference] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseReference].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReference] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrder] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrder]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrder] ( courtCaseSeizureOrderId, 
debtAmount, 
registrarUserId, 
dateUpdated, 
orderNo, 
attachedDate, 
movables, 
documentId, 
orderDate, 
immovables, 
isValidated, CourtCaseID )
 SELECT [DE_CourtCaseSeizureOrder].courtCaseSeizureOrderId, 
[DE_CourtCaseSeizureOrder].debtAmount, 
[DE_CourtCaseSeizureOrder].registrarUserId, 
[DE_CourtCaseSeizureOrder].dateUpdated, 
[DE_CourtCaseSeizureOrder].orderNo, 
[DE_CourtCaseSeizureOrder].attachedDate, 
[DE_CourtCaseSeizureOrder].movables, 
[DE_CourtCaseSeizureOrder].documentId, 
[DE_CourtCaseSeizureOrder].orderDate, 
[DE_CourtCaseSeizureOrder].immovables, 
[DE_CourtCaseSeizureOrder].isValidated, CourtCaseInstanceID
 FROM [DE_CourtCaseSeizureOrder] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSeizureOrder].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrder] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrderParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrderParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrderParty] ( courtCaseSeizureOrderPartyId, 
courtCaseSeizureOrderId, 
partyRoleId, 
partyID, 
isImmovablesOwner, 
isSecuringImmovables )
 SELECT [DE_CourtCaseSeizureOrderParty].courtCaseSeizureOrderPartyId, 
[DE_CourtCaseSeizureOrderParty].courtCaseSeizureOrderId, 
[DE_CourtCaseSeizureOrderParty].partyRoleId, 
[DE_CourtCaseSeizureOrderParty].partyInstanceId, 
[DE_CourtCaseSeizureOrderParty].isImmovablesOwner, 
[DE_CourtCaseSeizureOrderParty].isSecuringImmovables
 FROM [DE_CourtCaseSeizureOrderParty] INNER JOIN [DE_CourtCaseSeizureOrder] ON [DE_CourtCaseSeizureOrderParty].courtCaseSeizureOrderId = [DE_CourtCaseSeizureOrder].courtCaseSeizureOrderId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSeizureOrder].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizureOrderParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseHearingAdjournment] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseHearingAdjournment]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseHearingAdjournment] ( hearingTypeId, 
nextHearingDate, 
comments, 
adjournmentReasonId, 
courtCaseHearingAdjournmentId, 
canceledHearingDate, CourtCaseID )
 SELECT [DE_CourtCaseHearingAdjournment].hearingTypeId, 
[DE_CourtCaseHearingAdjournment].nextHearingDate, 
[DE_CourtCaseHearingAdjournment].comments, 
[DE_CourtCaseHearingAdjournment].adjournmentReasonId, 
[DE_CourtCaseHearingAdjournment].courtCaseHearingAdjournmentId, 
[DE_CourtCaseHearingAdjournment].canceledHearingDate, CourtCaseInstanceID
 FROM [DE_CourtCaseHearingAdjournment] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseHearingAdjournment].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseHearingAdjournment] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseComment] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseComment]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseComment] ( courtCaseCommentId, 
createdUserId, 
noteTypeId, 
createdDate, 
subject, CourtCaseID )
 SELECT [DE_CourtCaseComment].courtCaseCommentId, 
[DE_CourtCaseComment].createdUserId, 
[DE_CourtCaseComment].noteTypeId, 
[DE_CourtCaseComment].createdDate, 
[DE_CourtCaseComment].subject, CourtCaseInstanceID
 FROM [DE_CourtCaseComment] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseComment].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseComment] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetail] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetail]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetail] ( documentId, 
createdUserId, 
courtCaseCommentId, 
courtCaseCommentDetailId, 
createdDate, 
details )
 SELECT [DE_CourtCaseCommentDetail].documentId, 
[DE_CourtCaseCommentDetail].createdUserId, 
[DE_CourtCaseCommentDetail].courtCaseCommentId, 
[DE_CourtCaseCommentDetail].courtCaseCommentDetailId, 
[DE_CourtCaseCommentDetail].createdDate, 
[DE_CourtCaseCommentDetail].details
 FROM [DE_CourtCaseCommentDetail] INNER JOIN [DE_CourtCaseComment] ON [DE_CourtCaseCommentDetail].courtCaseCommentId = [DE_CourtCaseComment].courtCaseCommentId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseComment].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetail] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetailUser] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetailUser]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetailUser] ( userId, 
institutionId, 
courtCaseCommentId, 
courtCaseCommentDetailUserId )
 SELECT [DE_CourtCaseCommentDetailUser].userId, 
[DE_CourtCaseCommentDetailUser].institutionId, 
[DE_CourtCaseCommentDetailUser].courtCaseCommentId, 
[DE_CourtCaseCommentDetailUser].courtCaseCommentDetailUserId
 FROM [DE_CourtCaseCommentDetailUser] INNER JOIN [DE_CourtCaseComment] ON [DE_CourtCaseCommentDetailUser].courtCaseCommentId = [DE_CourtCaseComment].courtCaseCommentId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseComment].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCommentDetailUser] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysed] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysed]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysed] ( isSubmitToTheJudge, 
issue, 
dateUpdated, 
courtCaseIssuesToBeAnalysedId, 
description, 
registrarComment, 
updatedUserId, CourtCaseID )
 SELECT [DE_CourtCaseIssuesToBeAnalysed].isSubmitToTheJudge, 
[DE_CourtCaseIssuesToBeAnalysed].issue, 
[DE_CourtCaseIssuesToBeAnalysed].dateUpdated, 
[DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId, 
[DE_CourtCaseIssuesToBeAnalysed].description, 
[DE_CourtCaseIssuesToBeAnalysed].registrarComment, 
[DE_CourtCaseIssuesToBeAnalysed].updatedUserId, CourtCaseInstanceID
 FROM [DE_CourtCaseIssuesToBeAnalysed] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysed] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearing] ( courtCaseIssuesToBeAnalysedId, 
courtCaseIssuesToBeAnalysedJudgeHearingId, 
judgeComment, 
hearingDate )
 SELECT [DE_CourtCaseIssuesToBeAnalysedJudgeHearing].courtCaseIssuesToBeAnalysedId, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearing].courtCaseIssuesToBeAnalysedJudgeHearingId, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearing].judgeComment, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearing].hearingDate
 FROM [DE_CourtCaseIssuesToBeAnalysedJudgeHearing] INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedJudgeHearing].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearingQuestion] ( questionOrder, 
courtCaseIssuesToBeAnalysedJudgeHearingId, 
partyRoleId, 
partyID, 
question, 
representativePartyID, 
courtCaseIssuesToBeAnalysedJudgeHearingQuestionId )
 SELECT [DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].questionOrder, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].courtCaseIssuesToBeAnalysedJudgeHearingId, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].partyRoleId, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].partyInstanceId, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].question, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].courtCaseIssuesToBeAnalysedJudgeHearingQuestionId
 FROM [DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion] INNER JOIN [DE_CourtCaseIssuesToBeAnalysedJudgeHearing] ON [DE_CourtCaseIssuesToBeAnalysedJudgeHearingQuestion].courtCaseIssuesToBeAnalysedJudgeHearingId = [DE_CourtCaseIssuesToBeAnalysedJudgeHearing].courtCaseIssuesToBeAnalysedJudgeHearingId INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedJudgeHearing].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedJudgeHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedParty] ( statutes, 
legalCommentary, 
courtCaseIssuesToBeAnalysedId, 
isValidated, 
courtLegalPosition, 
courtCaseIssuesToBeAnalysedPartyId, 
updatedUserId, 
dateUpdated, 
dummyPartyId, 
descriptionOfFacts, 
partyID, 
precedent )
 SELECT [DE_CourtCaseIssuesToBeAnalysedParty].statutes, 
[DE_CourtCaseIssuesToBeAnalysedParty].legalCommentary, 
[DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedId, 
[DE_CourtCaseIssuesToBeAnalysedParty].isValidated, 
[DE_CourtCaseIssuesToBeAnalysedParty].courtLegalPosition, 
[DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedPartyId, 
[DE_CourtCaseIssuesToBeAnalysedParty].updatedUserId, 
[DE_CourtCaseIssuesToBeAnalysedParty].dateUpdated, 
[DE_CourtCaseIssuesToBeAnalysedParty].dummyPartyId, 
[DE_CourtCaseIssuesToBeAnalysedParty].descriptionOfFacts, 
[DE_CourtCaseIssuesToBeAnalysedParty].partyInstanceId, 
[DE_CourtCaseIssuesToBeAnalysedParty].precedent
 FROM [DE_CourtCaseIssuesToBeAnalysedParty] INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyDocument] ( details, 
courtCaseIssuesToBeAnalysedPartyDocumentId, 
courtCaseIssuesToBeAnalysedPartyId, 
documentId )
 SELECT [DE_CourtCaseIssuesToBeAnalysedPartyDocument].details, 
[DE_CourtCaseIssuesToBeAnalysedPartyDocument].courtCaseIssuesToBeAnalysedPartyDocumentId, 
[DE_CourtCaseIssuesToBeAnalysedPartyDocument].courtCaseIssuesToBeAnalysedPartyId, 
[DE_CourtCaseIssuesToBeAnalysedPartyDocument].documentId
 FROM [DE_CourtCaseIssuesToBeAnalysedPartyDocument] INNER JOIN [DE_CourtCaseIssuesToBeAnalysedParty] ON [DE_CourtCaseIssuesToBeAnalysedPartyDocument].courtCaseIssuesToBeAnalysedPartyId = [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedPartyId INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttle] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttle]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttle] ( courtCaseIssuesToBeAnalysedPartyRebuttleId, 
updatedUserId, 
descriptionOfFacts, 
precedent, 
legalCommentary, 
dateUpdated, 
partyID, 
courtLegalPosition, 
statutes, 
isValidated, 
courtCaseIssuesToBeAnalysedPartyId, 
dummyPartyId )
 SELECT [DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].courtCaseIssuesToBeAnalysedPartyRebuttleId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].updatedUserId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].descriptionOfFacts, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].precedent, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].legalCommentary, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].dateUpdated, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].partyInstanceId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].courtLegalPosition, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].statutes, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].isValidated, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].courtCaseIssuesToBeAnalysedPartyId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].dummyPartyId
 FROM [DE_CourtCaseIssuesToBeAnalysedPartyRebuttle] INNER JOIN [DE_CourtCaseIssuesToBeAnalysedParty] ON [DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].courtCaseIssuesToBeAnalysedPartyId = [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedPartyId INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttle] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttleDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttleDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttleDocument] ( courtCaseIssuesToBeAnalysedPartyRebuttleId, 
documentId, 
courtCaseIssuesToBeAnalysedPartyRebuttleDocumentId, 
details )
 SELECT [DE_CourtCaseIssuesToBeAnalysedPartyRebuttleDocument].courtCaseIssuesToBeAnalysedPartyRebuttleId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttleDocument].documentId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttleDocument].courtCaseIssuesToBeAnalysedPartyRebuttleDocumentId, 
[DE_CourtCaseIssuesToBeAnalysedPartyRebuttleDocument].details
 FROM [DE_CourtCaseIssuesToBeAnalysedPartyRebuttleDocument] INNER JOIN [DE_CourtCaseIssuesToBeAnalysedPartyRebuttle] ON [DE_CourtCaseIssuesToBeAnalysedPartyRebuttleDocument].courtCaseIssuesToBeAnalysedPartyRebuttleId = [DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].courtCaseIssuesToBeAnalysedPartyRebuttleId INNER JOIN [DE_CourtCaseIssuesToBeAnalysedParty] ON [DE_CourtCaseIssuesToBeAnalysedPartyRebuttle].courtCaseIssuesToBeAnalysedPartyId = [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedPartyId INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedParty].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedPartyRebuttleDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearing] ( courtCaseIssuesToBeAnalysedHearingId, 
hearingDate, 
courtCaseIssuesToBeAnalysedId )
 SELECT [DE_CourtCaseIssuesToBeAnalysedHearing].courtCaseIssuesToBeAnalysedHearingId, 
[DE_CourtCaseIssuesToBeAnalysedHearing].hearingDate, 
[DE_CourtCaseIssuesToBeAnalysedHearing].courtCaseIssuesToBeAnalysedId
 FROM [DE_CourtCaseIssuesToBeAnalysedHearing] INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedHearing].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearingQuestion] ( partyID, 
courtCaseIssuesToBeAnalysedHearingId, 
questionOrder, 
courtCaseIssuesToBeAnalysedHearingQuestionId, 
question, 
answer, 
partyRoleId, 
representativePartyID )
 SELECT [DE_CourtCaseIssuesToBeAnalysedHearingQuestion].partyInstanceId, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].courtCaseIssuesToBeAnalysedHearingId, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].questionOrder, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].courtCaseIssuesToBeAnalysedHearingQuestionId, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].question, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].answer, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].partyRoleId, 
[DE_CourtCaseIssuesToBeAnalysedHearingQuestion].representativePartyInstanceId
 FROM [DE_CourtCaseIssuesToBeAnalysedHearingQuestion] INNER JOIN [DE_CourtCaseIssuesToBeAnalysedHearing] ON [DE_CourtCaseIssuesToBeAnalysedHearingQuestion].courtCaseIssuesToBeAnalysedHearingId = [DE_CourtCaseIssuesToBeAnalysedHearing].courtCaseIssuesToBeAnalysedHearingId INNER JOIN [DE_CourtCaseIssuesToBeAnalysed] ON [DE_CourtCaseIssuesToBeAnalysedHearing].courtCaseIssuesToBeAnalysedId = [DE_CourtCaseIssuesToBeAnalysed].courtCaseIssuesToBeAnalysedId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIssuesToBeAnalysed].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIssuesToBeAnalysedHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFees] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFees]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFees] ( dateUpdated, 
updatedUserId, 
isSubmitToTheJudge, 
description, 
registrarComment, 
courtCaseProceduralFeesId, 
proceduralFees, CourtCaseID )
 SELECT [DE_CourtCaseProceduralFees].dateUpdated, 
[DE_CourtCaseProceduralFees].updatedUserId, 
[DE_CourtCaseProceduralFees].isSubmitToTheJudge, 
[DE_CourtCaseProceduralFees].description, 
[DE_CourtCaseProceduralFees].registrarComment, 
[DE_CourtCaseProceduralFees].courtCaseProceduralFeesId, 
[DE_CourtCaseProceduralFees].proceduralFees, CourtCaseInstanceID
 FROM [DE_CourtCaseProceduralFees] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFees] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesParty] ( isValidated, 
courtCaseProceduralFeesId, 
statutes, 
courtLegalPosition, 
courtCaseProceduralFeesPartyId, 
dummyPartyId, 
partyID, 
descriptionOfFacts, 
legalCommentary, 
updatedUserId, 
dateUpdated, 
precedent )
 SELECT [DE_CourtCaseProceduralFeesParty].isValidated, 
[DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesId, 
[DE_CourtCaseProceduralFeesParty].statutes, 
[DE_CourtCaseProceduralFeesParty].courtLegalPosition, 
[DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesPartyId, 
[DE_CourtCaseProceduralFeesParty].dummyPartyId, 
[DE_CourtCaseProceduralFeesParty].partyInstanceId, 
[DE_CourtCaseProceduralFeesParty].descriptionOfFacts, 
[DE_CourtCaseProceduralFeesParty].legalCommentary, 
[DE_CourtCaseProceduralFeesParty].updatedUserId, 
[DE_CourtCaseProceduralFeesParty].dateUpdated, 
[DE_CourtCaseProceduralFeesParty].precedent
 FROM [DE_CourtCaseProceduralFeesParty] INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttle] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttle]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttle] ( descriptionOfFacts, 
partyID, 
legalCommentary, 
statutes, 
dateUpdated, 
dummyPartyId, 
courtCaseProceduralFeesPartyRebuttleId, 
isValidated, 
precedent, 
courtLegalPosition, 
updatedUserId, 
courtCaseProceduralFeesPartyId )
 SELECT [DE_CourtCaseProceduralFeesPartyRebuttle].descriptionOfFacts, 
[DE_CourtCaseProceduralFeesPartyRebuttle].partyInstanceId, 
[DE_CourtCaseProceduralFeesPartyRebuttle].legalCommentary, 
[DE_CourtCaseProceduralFeesPartyRebuttle].statutes, 
[DE_CourtCaseProceduralFeesPartyRebuttle].dateUpdated, 
[DE_CourtCaseProceduralFeesPartyRebuttle].dummyPartyId, 
[DE_CourtCaseProceduralFeesPartyRebuttle].courtCaseProceduralFeesPartyRebuttleId, 
[DE_CourtCaseProceduralFeesPartyRebuttle].isValidated, 
[DE_CourtCaseProceduralFeesPartyRebuttle].precedent, 
[DE_CourtCaseProceduralFeesPartyRebuttle].courtLegalPosition, 
[DE_CourtCaseProceduralFeesPartyRebuttle].updatedUserId, 
[DE_CourtCaseProceduralFeesPartyRebuttle].courtCaseProceduralFeesPartyId
 FROM [DE_CourtCaseProceduralFeesPartyRebuttle] INNER JOIN [DE_CourtCaseProceduralFeesParty] ON [DE_CourtCaseProceduralFeesPartyRebuttle].courtCaseProceduralFeesPartyId = [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesPartyId INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttle] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttleDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttleDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttleDocument] ( courtCaseProceduralFeesPartyRebuttleId, 
details, 
documentId, 
courtCaseProceduralFeesPartyRebuttleDocumentId )
 SELECT [DE_CourtCaseProceduralFeesPartyRebuttleDocument].courtCaseProceduralFeesPartyRebuttleId, 
[DE_CourtCaseProceduralFeesPartyRebuttleDocument].details, 
[DE_CourtCaseProceduralFeesPartyRebuttleDocument].documentId, 
[DE_CourtCaseProceduralFeesPartyRebuttleDocument].courtCaseProceduralFeesPartyRebuttleDocumentId
 FROM [DE_CourtCaseProceduralFeesPartyRebuttleDocument] INNER JOIN [DE_CourtCaseProceduralFeesPartyRebuttle] ON [DE_CourtCaseProceduralFeesPartyRebuttleDocument].courtCaseProceduralFeesPartyRebuttleId = [DE_CourtCaseProceduralFeesPartyRebuttle].courtCaseProceduralFeesPartyRebuttleId INNER JOIN [DE_CourtCaseProceduralFeesParty] ON [DE_CourtCaseProceduralFeesPartyRebuttle].courtCaseProceduralFeesPartyId = [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesPartyId INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyRebuttleDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyDocument] ( courtCaseProceduralFeesPartyDocumentId, 
documentId, 
courtCaseProceduralFeesPartyId, 
details )
 SELECT [DE_CourtCaseProceduralFeesPartyDocument].courtCaseProceduralFeesPartyDocumentId, 
[DE_CourtCaseProceduralFeesPartyDocument].documentId, 
[DE_CourtCaseProceduralFeesPartyDocument].courtCaseProceduralFeesPartyId, 
[DE_CourtCaseProceduralFeesPartyDocument].details
 FROM [DE_CourtCaseProceduralFeesPartyDocument] INNER JOIN [DE_CourtCaseProceduralFeesParty] ON [DE_CourtCaseProceduralFeesPartyDocument].courtCaseProceduralFeesPartyId = [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesPartyId INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesParty].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesPartyDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearing] ( courtCaseProceduralFeesId, 
hearingDate, 
courtCaseProceduralFeesJudgeHearingId, 
judgeComment )
 SELECT [DE_CourtCaseProceduralFeesJudgeHearing].courtCaseProceduralFeesId, 
[DE_CourtCaseProceduralFeesJudgeHearing].hearingDate, 
[DE_CourtCaseProceduralFeesJudgeHearing].courtCaseProceduralFeesJudgeHearingId, 
[DE_CourtCaseProceduralFeesJudgeHearing].judgeComment
 FROM [DE_CourtCaseProceduralFeesJudgeHearing] INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesJudgeHearing].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearingQuestion] ( courtCaseProceduralFeesJudgeHearingQuestionId, 
partyRoleId, 
partyID, 
courtCaseProceduralFeesJudgeHearingId, 
questionOrder, 
question, 
representativePartyID )
 SELECT [DE_CourtCaseProceduralFeesJudgeHearingQuestion].courtCaseProceduralFeesJudgeHearingQuestionId, 
[DE_CourtCaseProceduralFeesJudgeHearingQuestion].partyRoleId, 
[DE_CourtCaseProceduralFeesJudgeHearingQuestion].partyInstanceId, 
[DE_CourtCaseProceduralFeesJudgeHearingQuestion].courtCaseProceduralFeesJudgeHearingId, 
[DE_CourtCaseProceduralFeesJudgeHearingQuestion].questionOrder, 
[DE_CourtCaseProceduralFeesJudgeHearingQuestion].question, 
[DE_CourtCaseProceduralFeesJudgeHearingQuestion].representativePartyInstanceId
 FROM [DE_CourtCaseProceduralFeesJudgeHearingQuestion] INNER JOIN [DE_CourtCaseProceduralFeesJudgeHearing] ON [DE_CourtCaseProceduralFeesJudgeHearingQuestion].courtCaseProceduralFeesJudgeHearingId = [DE_CourtCaseProceduralFeesJudgeHearing].courtCaseProceduralFeesJudgeHearingId INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesJudgeHearing].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesJudgeHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearing] ( hearingDate, 
courtCaseProceduralFeesId, 
courtCaseProceduralFeesHearingId )
 SELECT [DE_CourtCaseProceduralFeesHearing].hearingDate, 
[DE_CourtCaseProceduralFeesHearing].courtCaseProceduralFeesId, 
[DE_CourtCaseProceduralFeesHearing].courtCaseProceduralFeesHearingId
 FROM [DE_CourtCaseProceduralFeesHearing] INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesHearing].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearingQuestion] ( questionOrder, 
answer, 
question, 
partyRoleId, 
representativePartyID, 
partyID, 
courtCaseProceduralFeesHearingId, 
courtCaseProceduralFeesHearingQuestionId )
 SELECT [DE_CourtCaseProceduralFeesHearingQuestion].questionOrder, 
[DE_CourtCaseProceduralFeesHearingQuestion].answer, 
[DE_CourtCaseProceduralFeesHearingQuestion].question, 
[DE_CourtCaseProceduralFeesHearingQuestion].partyRoleId, 
[DE_CourtCaseProceduralFeesHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseProceduralFeesHearingQuestion].partyInstanceId, 
[DE_CourtCaseProceduralFeesHearingQuestion].courtCaseProceduralFeesHearingId, 
[DE_CourtCaseProceduralFeesHearingQuestion].courtCaseProceduralFeesHearingQuestionId
 FROM [DE_CourtCaseProceduralFeesHearingQuestion] INNER JOIN [DE_CourtCaseProceduralFeesHearing] ON [DE_CourtCaseProceduralFeesHearingQuestion].courtCaseProceduralFeesHearingId = [DE_CourtCaseProceduralFeesHearing].courtCaseProceduralFeesHearingId INNER JOIN [DE_CourtCaseProceduralFees] ON [DE_CourtCaseProceduralFeesHearing].courtCaseProceduralFeesId = [DE_CourtCaseProceduralFees].courtCaseProceduralFeesId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceduralFees].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceduralFeesHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOtherFee] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOtherFee]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOtherFee] ( soldPages, 
attachedDate, 
paidAmount, 
receiptNumber, 
paymentDate, 
courtCaseOtherFeeId, 
paymentBankId, 
documentId, 
details, 
paymentTypeId, 
isValidated, CourtCaseID )
 SELECT [DE_CourtCaseOtherFee].soldPages, 
[DE_CourtCaseOtherFee].attachedDate, 
[DE_CourtCaseOtherFee].paidAmount, 
[DE_CourtCaseOtherFee].receiptNumber, 
[DE_CourtCaseOtherFee].paymentDate, 
[DE_CourtCaseOtherFee].courtCaseOtherFeeId, 
[DE_CourtCaseOtherFee].paymentBankId, 
[DE_CourtCaseOtherFee].documentId, 
[DE_CourtCaseOtherFee].details, 
[DE_CourtCaseOtherFee].paymentTypeId, 
[DE_CourtCaseOtherFee].isValidated, CourtCaseInstanceID
 FROM [DE_CourtCaseOtherFee] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseOtherFee].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOtherFee] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceeding] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceeding]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceeding] ( reOpeningReasonId, 
isDatesObeyed, 
conclusion, 
dateUpdated, 
postponementReasonId, 
pronouncementDate, 
hearingTypeId, 
registrarUserId, 
interlocutoryJudgementId, 
isValidated, 
dateOfJudgment, 
offence, 
decision, 
proceedingTypeId, 
decisionNumber, 
nextHearingDate, 
apealDate, 
observation, 
isMediationProposal, 
courtCaseProceedingId, 
proceedingDate, 
hearingDate, CourtCaseID )
 SELECT [DE_CourtCaseProceeding].reOpeningReasonId, 
[DE_CourtCaseProceeding].isDatesObeyed, 
[DE_CourtCaseProceeding].conclusion, 
[DE_CourtCaseProceeding].dateUpdated, 
[DE_CourtCaseProceeding].postponementReasonId, 
[DE_CourtCaseProceeding].pronouncementDate, 
[DE_CourtCaseProceeding].hearingTypeId, 
[DE_CourtCaseProceeding].registrarUserId, 
[DE_CourtCaseProceeding].interlocutoryJudgementId, 
[DE_CourtCaseProceeding].isValidated, 
[DE_CourtCaseProceeding].dateOfJudgment, 
[DE_CourtCaseProceeding].offence, 
[DE_CourtCaseProceeding].decision, 
[DE_CourtCaseProceeding].proceedingTypeId, 
[DE_CourtCaseProceeding].decisionNumber, 
[DE_CourtCaseProceeding].nextHearingDate, 
[DE_CourtCaseProceeding].apealDate, 
[DE_CourtCaseProceeding].observation, 
[DE_CourtCaseProceeding].isMediationProposal, 
[DE_CourtCaseProceeding].courtCaseProceedingId, 
[DE_CourtCaseProceeding].proceedingDate, 
[DE_CourtCaseProceeding].hearingDate, CourtCaseInstanceID
 FROM [DE_CourtCaseProceeding] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceeding].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceeding] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingAttachment] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingAttachment]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingAttachment] ( courtCaseProceedingId, 
fileDate, 
documentId, 
courtCaseProceedingAttachmentId )
 SELECT [DE_CourtCaseProceedingAttachment].courtCaseProceedingId, 
[DE_CourtCaseProceedingAttachment].fileDate, 
[DE_CourtCaseProceedingAttachment].documentId, 
[DE_CourtCaseProceedingAttachment].courtCaseProceedingAttachmentId
 FROM [DE_CourtCaseProceedingAttachment] INNER JOIN [DE_CourtCaseProceeding] ON [DE_CourtCaseProceedingAttachment].courtCaseProceedingId = [DE_CourtCaseProceeding].courtCaseProceedingId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceeding].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingAttachment] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingQuestion] ( partyID, 
representativePartyID, 
courtCaseProceedingId, 
questionNo, 
question, 
answer, 
courtCaseProceedingQuestionId )
 SELECT [DE_CourtCaseProceedingQuestion].partyInstanceId, 
[DE_CourtCaseProceedingQuestion].representativePartyInstanceId, 
[DE_CourtCaseProceedingQuestion].courtCaseProceedingId, 
[DE_CourtCaseProceedingQuestion].questionNo, 
[DE_CourtCaseProceedingQuestion].question, 
[DE_CourtCaseProceedingQuestion].answer, 
[DE_CourtCaseProceedingQuestion].courtCaseProceedingQuestionId
 FROM [DE_CourtCaseProceedingQuestion] INNER JOIN [DE_CourtCaseProceeding] ON [DE_CourtCaseProceedingQuestion].courtCaseProceedingId = [DE_CourtCaseProceeding].courtCaseProceedingId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceeding].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingParty] ( courtCaseProceedingPartyId, 
partyRoleId, 
partyID, 
courtCaseProceedingId, 
isPresent )
 SELECT [DE_CourtCaseProceedingParty].courtCaseProceedingPartyId, 
[DE_CourtCaseProceedingParty].partyRoleId, 
[DE_CourtCaseProceedingParty].partyInstanceId, 
[DE_CourtCaseProceedingParty].courtCaseProceedingId, 
[DE_CourtCaseProceedingParty].isPresent
 FROM [DE_CourtCaseProceedingParty] INNER JOIN [DE_CourtCaseProceeding] ON [DE_CourtCaseProceedingParty].courtCaseProceedingId = [DE_CourtCaseProceeding].courtCaseProceedingId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseProceeding].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseProceedingParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummon] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummon]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummon] ( date, 
registrarUserId, 
post, 
phone, 
email, 
summonDate, 
isValidated, 
name, 
officialTellingToSign, 
dateUpdated, 
atCourtAppearDate, 
orderDate, 
documentId, 
courtCaseSummonId, 
metWithId, 
summonTypeId, 
orderNo, 
summonRequestDate, 
litigationObject, CourtCaseID )
 SELECT [DE_CourtCaseSummon].date, 
[DE_CourtCaseSummon].registrarUserId, 
[DE_CourtCaseSummon].post, 
[DE_CourtCaseSummon].phone, 
[DE_CourtCaseSummon].email, 
[DE_CourtCaseSummon].summonDate, 
[DE_CourtCaseSummon].isValidated, 
[DE_CourtCaseSummon].name, 
[DE_CourtCaseSummon].officialTellingToSign, 
[DE_CourtCaseSummon].dateUpdated, 
[DE_CourtCaseSummon].atCourtAppearDate, 
[DE_CourtCaseSummon].orderDate, 
[DE_CourtCaseSummon].documentId, 
[DE_CourtCaseSummon].courtCaseSummonId, 
[DE_CourtCaseSummon].metWithId, 
[DE_CourtCaseSummon].summonTypeId, 
[DE_CourtCaseSummon].orderNo, 
[DE_CourtCaseSummon].summonRequestDate, 
[DE_CourtCaseSummon].litigationObject, CourtCaseInstanceID
 FROM [DE_CourtCaseSummon] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSummon].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummon] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummonParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummonParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummonParty] ( isSummoned, 
courtCaseSummonId, 
isRequestor, 
partyRoleId, 
partyID, 
courtCaseSummonPartyId )
 SELECT [DE_CourtCaseSummonParty].isSummoned, 
[DE_CourtCaseSummonParty].courtCaseSummonId, 
[DE_CourtCaseSummonParty].isRequestor, 
[DE_CourtCaseSummonParty].partyRoleId, 
[DE_CourtCaseSummonParty].partyInstanceId, 
[DE_CourtCaseSummonParty].courtCaseSummonPartyId
 FROM [DE_CourtCaseSummonParty] INNER JOIN [DE_CourtCaseSummon] ON [DE_CourtCaseSummonParty].courtCaseSummonId = [DE_CourtCaseSummon].courtCaseSummonId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSummon].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSummonParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamage] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamage]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamage] ( courtCaseDamageId, 
updatedUserId, 
dateUpdated, 
damage, 
registrarComment, 
description, 
isSubmitToTheJudge, CourtCaseID )
 SELECT [DE_CourtCaseDamage].courtCaseDamageId, 
[DE_CourtCaseDamage].updatedUserId, 
[DE_CourtCaseDamage].dateUpdated, 
[DE_CourtCaseDamage].damage, 
[DE_CourtCaseDamage].registrarComment, 
[DE_CourtCaseDamage].description, 
[DE_CourtCaseDamage].isSubmitToTheJudge, CourtCaseInstanceID
 FROM [DE_CourtCaseDamage] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamage] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageParty] ( courtCaseDamagePartyId, 
isValidated, 
dummyPartyId, 
dateUpdated, 
updatedUserId, 
courtLegalPosition, 
partyID, 
courtCaseDamageId, 
statutes, 
precedent, 
descriptionOfFacts, 
legalCommentary )
 SELECT [DE_CourtCaseDamageParty].courtCaseDamagePartyId, 
[DE_CourtCaseDamageParty].isValidated, 
[DE_CourtCaseDamageParty].dummyPartyId, 
[DE_CourtCaseDamageParty].dateUpdated, 
[DE_CourtCaseDamageParty].updatedUserId, 
[DE_CourtCaseDamageParty].courtLegalPosition, 
[DE_CourtCaseDamageParty].partyInstanceId, 
[DE_CourtCaseDamageParty].courtCaseDamageId, 
[DE_CourtCaseDamageParty].statutes, 
[DE_CourtCaseDamageParty].precedent, 
[DE_CourtCaseDamageParty].descriptionOfFacts, 
[DE_CourtCaseDamageParty].legalCommentary
 FROM [DE_CourtCaseDamageParty] INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageParty].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyDocument] ( courtCaseDamagePartyDocumentId, 
courtCaseDamagePartyId, 
documentId, 
details )
 SELECT [DE_CourtCaseDamagePartyDocument].courtCaseDamagePartyDocumentId, 
[DE_CourtCaseDamagePartyDocument].courtCaseDamagePartyId, 
[DE_CourtCaseDamagePartyDocument].documentId, 
[DE_CourtCaseDamagePartyDocument].details
 FROM [DE_CourtCaseDamagePartyDocument] INNER JOIN [DE_CourtCaseDamageParty] ON [DE_CourtCaseDamagePartyDocument].courtCaseDamagePartyId = [DE_CourtCaseDamageParty].courtCaseDamagePartyId INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageParty].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttle] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttle]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttle] ( descriptionOfFacts, 
partyID, 
statutes, 
legalCommentary, 
courtCaseDamagePartyId, 
dummyPartyId, 
courtLegalPosition, 
precedent, 
dateUpdated, 
updatedUserId, 
courtCaseDamagePartyRebuttleId, 
isValidated )
 SELECT [DE_CourtCaseDamagePartyRebuttle].descriptionOfFacts, 
[DE_CourtCaseDamagePartyRebuttle].partyInstanceId, 
[DE_CourtCaseDamagePartyRebuttle].statutes, 
[DE_CourtCaseDamagePartyRebuttle].legalCommentary, 
[DE_CourtCaseDamagePartyRebuttle].courtCaseDamagePartyId, 
[DE_CourtCaseDamagePartyRebuttle].dummyPartyId, 
[DE_CourtCaseDamagePartyRebuttle].courtLegalPosition, 
[DE_CourtCaseDamagePartyRebuttle].precedent, 
[DE_CourtCaseDamagePartyRebuttle].dateUpdated, 
[DE_CourtCaseDamagePartyRebuttle].updatedUserId, 
[DE_CourtCaseDamagePartyRebuttle].courtCaseDamagePartyRebuttleId, 
[DE_CourtCaseDamagePartyRebuttle].isValidated
 FROM [DE_CourtCaseDamagePartyRebuttle] INNER JOIN [DE_CourtCaseDamageParty] ON [DE_CourtCaseDamagePartyRebuttle].courtCaseDamagePartyId = [DE_CourtCaseDamageParty].courtCaseDamagePartyId INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageParty].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttle] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttleDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttleDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttleDocument] ( courtCaseDamagePartyRebuttleDocumentId, 
details, 
courtCaseDamagePartyRebuttleId, 
documentId )
 SELECT [DE_CourtCaseDamagePartyRebuttleDocument].courtCaseDamagePartyRebuttleDocumentId, 
[DE_CourtCaseDamagePartyRebuttleDocument].details, 
[DE_CourtCaseDamagePartyRebuttleDocument].courtCaseDamagePartyRebuttleId, 
[DE_CourtCaseDamagePartyRebuttleDocument].documentId
 FROM [DE_CourtCaseDamagePartyRebuttleDocument] INNER JOIN [DE_CourtCaseDamagePartyRebuttle] ON [DE_CourtCaseDamagePartyRebuttleDocument].courtCaseDamagePartyRebuttleId = [DE_CourtCaseDamagePartyRebuttle].courtCaseDamagePartyRebuttleId INNER JOIN [DE_CourtCaseDamageParty] ON [DE_CourtCaseDamagePartyRebuttle].courtCaseDamagePartyId = [DE_CourtCaseDamageParty].courtCaseDamagePartyId INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageParty].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamagePartyRebuttleDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearing] ( judgeComment, 
courtCaseDamageId, 
courtCaseDamageJudgeHearingId, 
hearingDate )
 SELECT [DE_CourtCaseDamageJudgeHearing].judgeComment, 
[DE_CourtCaseDamageJudgeHearing].courtCaseDamageId, 
[DE_CourtCaseDamageJudgeHearing].courtCaseDamageJudgeHearingId, 
[DE_CourtCaseDamageJudgeHearing].hearingDate
 FROM [DE_CourtCaseDamageJudgeHearing] INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageJudgeHearing].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearingQuestion] ( courtCaseDamageJudgeHearingId, 
questionOrder, 
question, 
partyRoleId, 
partyID, 
representativePartyID, 
courtCaseDamageJudgeHearingQuestionId )
 SELECT [DE_CourtCaseDamageJudgeHearingQuestion].courtCaseDamageJudgeHearingId, 
[DE_CourtCaseDamageJudgeHearingQuestion].questionOrder, 
[DE_CourtCaseDamageJudgeHearingQuestion].question, 
[DE_CourtCaseDamageJudgeHearingQuestion].partyRoleId, 
[DE_CourtCaseDamageJudgeHearingQuestion].partyInstanceId, 
[DE_CourtCaseDamageJudgeHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseDamageJudgeHearingQuestion].courtCaseDamageJudgeHearingQuestionId
 FROM [DE_CourtCaseDamageJudgeHearingQuestion] INNER JOIN [DE_CourtCaseDamageJudgeHearing] ON [DE_CourtCaseDamageJudgeHearingQuestion].courtCaseDamageJudgeHearingId = [DE_CourtCaseDamageJudgeHearing].courtCaseDamageJudgeHearingId INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageJudgeHearing].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageJudgeHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearing] ( courtCaseDamageHearingId, 
hearingDate, 
courtCaseDamageId )
 SELECT [DE_CourtCaseDamageHearing].courtCaseDamageHearingId, 
[DE_CourtCaseDamageHearing].hearingDate, 
[DE_CourtCaseDamageHearing].courtCaseDamageId
 FROM [DE_CourtCaseDamageHearing] INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageHearing].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearingQuestion] ( representativePartyID, 
partyRoleId, 
courtCaseDamageHearingId, 
courtCaseDamageHearingQuestionId, 
question, 
answer, 
questionOrder, 
partyID )
 SELECT [DE_CourtCaseDamageHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseDamageHearingQuestion].partyRoleId, 
[DE_CourtCaseDamageHearingQuestion].courtCaseDamageHearingId, 
[DE_CourtCaseDamageHearingQuestion].courtCaseDamageHearingQuestionId, 
[DE_CourtCaseDamageHearingQuestion].question, 
[DE_CourtCaseDamageHearingQuestion].answer, 
[DE_CourtCaseDamageHearingQuestion].questionOrder, 
[DE_CourtCaseDamageHearingQuestion].partyInstanceId
 FROM [DE_CourtCaseDamageHearingQuestion] INNER JOIN [DE_CourtCaseDamageHearing] ON [DE_CourtCaseDamageHearingQuestion].courtCaseDamageHearingId = [DE_CourtCaseDamageHearing].courtCaseDamageHearingId INNER JOIN [DE_CourtCaseDamage] ON [DE_CourtCaseDamageHearing].courtCaseDamageId = [DE_CourtCaseDamage].courtCaseDamageId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDamage].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDamageHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwelling] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwelling]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwelling] ( courtCaseOrderKnownDwellingId, 
isValidated, 
attachedDate, 
governmentPaper, 
documentId, 
registrarUserId, 
putOnDate, 
orderNo, 
dateUpdated, 
orderDate, CourtCaseID )
 SELECT [DE_CourtCaseOrderKnownDwelling].courtCaseOrderKnownDwellingId, 
[DE_CourtCaseOrderKnownDwelling].isValidated, 
[DE_CourtCaseOrderKnownDwelling].attachedDate, 
[DE_CourtCaseOrderKnownDwelling].governmentPaper, 
[DE_CourtCaseOrderKnownDwelling].documentId, 
[DE_CourtCaseOrderKnownDwelling].registrarUserId, 
[DE_CourtCaseOrderKnownDwelling].putOnDate, 
[DE_CourtCaseOrderKnownDwelling].orderNo, 
[DE_CourtCaseOrderKnownDwelling].dateUpdated, 
[DE_CourtCaseOrderKnownDwelling].orderDate, CourtCaseInstanceID
 FROM [DE_CourtCaseOrderKnownDwelling] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseOrderKnownDwelling].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwelling] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwellingParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwellingParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwellingParty] ( courtCaseOrderKnownDwellingPartyId, 
partyID, 
courtCaseOrderKnownDwellingId, 
partyRoleId )
 SELECT [DE_CourtCaseOrderKnownDwellingParty].courtCaseOrderKnownDwellingPartyId, 
[DE_CourtCaseOrderKnownDwellingParty].partyInstanceId, 
[DE_CourtCaseOrderKnownDwellingParty].courtCaseOrderKnownDwellingId, 
[DE_CourtCaseOrderKnownDwellingParty].partyRoleId
 FROM [DE_CourtCaseOrderKnownDwellingParty] INNER JOIN [DE_CourtCaseOrderKnownDwelling] ON [DE_CourtCaseOrderKnownDwellingParty].courtCaseOrderKnownDwellingId = [DE_CourtCaseOrderKnownDwelling].courtCaseOrderKnownDwellingId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseOrderKnownDwelling].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOrderKnownDwellingParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItem] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItem]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItem] ( isSubmitToTheJudge, 
dateUpdated, 
courtCaseAdmissibilityItemId, 
registrarComment, 
admissibilityItemId, 
description, 
updatedUserId, 
otherAdmissibilityItem, CourtCaseID )
 SELECT [DE_CourtCaseAdmissibilityItem].isSubmitToTheJudge, 
[DE_CourtCaseAdmissibilityItem].dateUpdated, 
[DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId, 
[DE_CourtCaseAdmissibilityItem].registrarComment, 
[DE_CourtCaseAdmissibilityItem].admissibilityItemId, 
[DE_CourtCaseAdmissibilityItem].description, 
[DE_CourtCaseAdmissibilityItem].updatedUserId, 
[DE_CourtCaseAdmissibilityItem].otherAdmissibilityItem, CourtCaseInstanceID
 FROM [DE_CourtCaseAdmissibilityItem] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItem] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearing] ( courtCaseAdmissibilityItemId, 
hearingDate, 
courtCaseAdmissibilityItemHearingId )
 SELECT [DE_CourtCaseAdmissibilityItemHearing].courtCaseAdmissibilityItemId, 
[DE_CourtCaseAdmissibilityItemHearing].hearingDate, 
[DE_CourtCaseAdmissibilityItemHearing].courtCaseAdmissibilityItemHearingId
 FROM [DE_CourtCaseAdmissibilityItemHearing] INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemHearing].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearingQuestion] ( courtCaseAdmissibilityItemHearingId, 
question, 
answer, 
courtCaseAdmissibilityItemHearingQuestionId, 
representativePartyID, 
partyID, 
questionOrder, 
partyRoleId )
 SELECT [DE_CourtCaseAdmissibilityItemHearingQuestion].courtCaseAdmissibilityItemHearingId, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].question, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].answer, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].courtCaseAdmissibilityItemHearingQuestionId, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].partyInstanceId, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].questionOrder, 
[DE_CourtCaseAdmissibilityItemHearingQuestion].partyRoleId
 FROM [DE_CourtCaseAdmissibilityItemHearingQuestion] INNER JOIN [DE_CourtCaseAdmissibilityItemHearing] ON [DE_CourtCaseAdmissibilityItemHearingQuestion].courtCaseAdmissibilityItemHearingId = [DE_CourtCaseAdmissibilityItemHearing].courtCaseAdmissibilityItemHearingId INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemHearing].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemParty] ( courtLegalPosition, 
isValidated, 
statutes, 
precedent, 
dateUpdated, 
descriptionOfFacts, 
courtCaseAdmissibilityItemPartyId, 
partyID, 
updatedUserId, 
legalCommentary, 
dummyPartyId, 
courtCaseAdmissibilityItemId )
 SELECT [DE_CourtCaseAdmissibilityItemParty].courtLegalPosition, 
[DE_CourtCaseAdmissibilityItemParty].isValidated, 
[DE_CourtCaseAdmissibilityItemParty].statutes, 
[DE_CourtCaseAdmissibilityItemParty].precedent, 
[DE_CourtCaseAdmissibilityItemParty].dateUpdated, 
[DE_CourtCaseAdmissibilityItemParty].descriptionOfFacts, 
[DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemPartyId, 
[DE_CourtCaseAdmissibilityItemParty].partyInstanceId, 
[DE_CourtCaseAdmissibilityItemParty].updatedUserId, 
[DE_CourtCaseAdmissibilityItemParty].legalCommentary, 
[DE_CourtCaseAdmissibilityItemParty].dummyPartyId, 
[DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemId
 FROM [DE_CourtCaseAdmissibilityItemParty] INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyDocument] ( courtCaseAdmissibilityItemPartyId, 
courtCaseAdmissibilityItemPartyDocumentId, 
details, 
documentId )
 SELECT [DE_CourtCaseAdmissibilityItemPartyDocument].courtCaseAdmissibilityItemPartyId, 
[DE_CourtCaseAdmissibilityItemPartyDocument].courtCaseAdmissibilityItemPartyDocumentId, 
[DE_CourtCaseAdmissibilityItemPartyDocument].details, 
[DE_CourtCaseAdmissibilityItemPartyDocument].documentId
 FROM [DE_CourtCaseAdmissibilityItemPartyDocument] INNER JOIN [DE_CourtCaseAdmissibilityItemParty] ON [DE_CourtCaseAdmissibilityItemPartyDocument].courtCaseAdmissibilityItemPartyId = [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemPartyId INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttle] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttle]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttle] ( legalCommentary, 
courtCaseAdmissibilityItemPartyId, 
partyID, 
courtCaseAdmissibilityItemPartyRebuttleId, 
descriptionOfFacts, 
statutes, 
dateUpdated, 
precedent, 
dummyPartyId, 
courtLegalPosition, 
isValidated, 
updatedUserId )
 SELECT [DE_CourtCaseAdmissibilityItemPartyRebuttle].legalCommentary, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].courtCaseAdmissibilityItemPartyId, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].partyInstanceId, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].courtCaseAdmissibilityItemPartyRebuttleId, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].descriptionOfFacts, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].statutes, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].dateUpdated, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].precedent, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].dummyPartyId, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].courtLegalPosition, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].isValidated, 
[DE_CourtCaseAdmissibilityItemPartyRebuttle].updatedUserId
 FROM [DE_CourtCaseAdmissibilityItemPartyRebuttle] INNER JOIN [DE_CourtCaseAdmissibilityItemParty] ON [DE_CourtCaseAdmissibilityItemPartyRebuttle].courtCaseAdmissibilityItemPartyId = [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemPartyId INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttle] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttleDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttleDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttleDocument] ( courtCaseAdmissibilityItemPartyRebuttleDocumentId, 
details, 
courtCaseAdmissibilityItemPartyRebuttleId, 
documentId )
 SELECT [DE_CourtCaseAdmissibilityItemPartyRebuttleDocument].courtCaseAdmissibilityItemPartyRebuttleDocumentId, 
[DE_CourtCaseAdmissibilityItemPartyRebuttleDocument].details, 
[DE_CourtCaseAdmissibilityItemPartyRebuttleDocument].courtCaseAdmissibilityItemPartyRebuttleId, 
[DE_CourtCaseAdmissibilityItemPartyRebuttleDocument].documentId
 FROM [DE_CourtCaseAdmissibilityItemPartyRebuttleDocument] INNER JOIN [DE_CourtCaseAdmissibilityItemPartyRebuttle] ON [DE_CourtCaseAdmissibilityItemPartyRebuttleDocument].courtCaseAdmissibilityItemPartyRebuttleId = [DE_CourtCaseAdmissibilityItemPartyRebuttle].courtCaseAdmissibilityItemPartyRebuttleId INNER JOIN [DE_CourtCaseAdmissibilityItemParty] ON [DE_CourtCaseAdmissibilityItemPartyRebuttle].courtCaseAdmissibilityItemPartyId = [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemPartyId INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemParty].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemPartyRebuttleDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearing] ( courtCaseAdmissibilityItemJudgeHearingId, 
hearingDate, 
judgeComment, 
courtCaseAdmissibilityItemId )
 SELECT [DE_CourtCaseAdmissibilityItemJudgeHearing].courtCaseAdmissibilityItemJudgeHearingId, 
[DE_CourtCaseAdmissibilityItemJudgeHearing].hearingDate, 
[DE_CourtCaseAdmissibilityItemJudgeHearing].judgeComment, 
[DE_CourtCaseAdmissibilityItemJudgeHearing].courtCaseAdmissibilityItemId
 FROM [DE_CourtCaseAdmissibilityItemJudgeHearing] INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemJudgeHearing].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearingQuestion] ( questionOrder, 
courtCaseAdmissibilityItemJudgeHearingId, 
partyID, 
representativePartyID, 
partyRoleId, 
courtCaseAdmissibilityItemJudgeHearingQuestionId, 
question )
 SELECT [DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].questionOrder, 
[DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].courtCaseAdmissibilityItemJudgeHearingId, 
[DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].partyInstanceId, 
[DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].partyRoleId, 
[DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].courtCaseAdmissibilityItemJudgeHearingQuestionId, 
[DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].question
 FROM [DE_CourtCaseAdmissibilityItemJudgeHearingQuestion] INNER JOIN [DE_CourtCaseAdmissibilityItemJudgeHearing] ON [DE_CourtCaseAdmissibilityItemJudgeHearingQuestion].courtCaseAdmissibilityItemJudgeHearingId = [DE_CourtCaseAdmissibilityItemJudgeHearing].courtCaseAdmissibilityItemJudgeHearingId INNER JOIN [DE_CourtCaseAdmissibilityItem] ON [DE_CourtCaseAdmissibilityItemJudgeHearing].courtCaseAdmissibilityItemId = [DE_CourtCaseAdmissibilityItem].courtCaseAdmissibilityItemId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseAdmissibilityItem].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseAdmissibilityItemJudgeHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseBenchDetail] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseBenchDetail]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseBenchDetail] ( isPresident, 
courtCaseBenchDetailId, 
judgeUserId, CourtCaseID )
 SELECT [DE_CourtCaseBenchDetail].isPresident, 
[DE_CourtCaseBenchDetail].courtCaseBenchDetailId, 
[DE_CourtCaseBenchDetail].judgeUserId, CourtCaseInstanceID
 FROM [DE_CourtCaseBenchDetail] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseBenchDetail].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseBenchDetail] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaim] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaim]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaim] ( updatedUserId, 
courtCaseCounterClaimId, 
counterClaim, 
isSubmitToTheJudge, 
description, 
registrarComment, 
dateUpdated, CourtCaseID )
 SELECT [DE_CourtCaseCounterClaim].updatedUserId, 
[DE_CourtCaseCounterClaim].courtCaseCounterClaimId, 
[DE_CourtCaseCounterClaim].counterClaim, 
[DE_CourtCaseCounterClaim].isSubmitToTheJudge, 
[DE_CourtCaseCounterClaim].description, 
[DE_CourtCaseCounterClaim].registrarComment, 
[DE_CourtCaseCounterClaim].dateUpdated, CourtCaseInstanceID
 FROM [DE_CourtCaseCounterClaim] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaim] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimParty] ( statutes, 
legalCommentary, 
courtCaseCounterClaimPartyId, 
descriptionOfFacts, 
partyID, 
precedent, 
courtCaseCounterClaimId, 
dateUpdated, 
updatedUserId, 
dummyPartyId, 
courtLegalPosition, 
isValidated )
 SELECT [DE_CourtCaseCounterClaimParty].statutes, 
[DE_CourtCaseCounterClaimParty].legalCommentary, 
[DE_CourtCaseCounterClaimParty].courtCaseCounterClaimPartyId, 
[DE_CourtCaseCounterClaimParty].descriptionOfFacts, 
[DE_CourtCaseCounterClaimParty].partyInstanceId, 
[DE_CourtCaseCounterClaimParty].precedent, 
[DE_CourtCaseCounterClaimParty].courtCaseCounterClaimId, 
[DE_CourtCaseCounterClaimParty].dateUpdated, 
[DE_CourtCaseCounterClaimParty].updatedUserId, 
[DE_CourtCaseCounterClaimParty].dummyPartyId, 
[DE_CourtCaseCounterClaimParty].courtLegalPosition, 
[DE_CourtCaseCounterClaimParty].isValidated
 FROM [DE_CourtCaseCounterClaimParty] INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttle] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttle]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttle] ( legalCommentary, 
partyID, 
descriptionOfFacts, 
statutes, 
precedent, 
dateUpdated, 
dummyPartyId, 
courtCaseCounterClaimPartyRebuttleId, 
courtLegalPosition, 
courtCaseCounterClaimPartyId, 
isValidated, 
updatedUserId )
 SELECT [DE_CourtCaseCounterClaimPartyRebuttle].legalCommentary, 
[DE_CourtCaseCounterClaimPartyRebuttle].partyInstanceId, 
[DE_CourtCaseCounterClaimPartyRebuttle].descriptionOfFacts, 
[DE_CourtCaseCounterClaimPartyRebuttle].statutes, 
[DE_CourtCaseCounterClaimPartyRebuttle].precedent, 
[DE_CourtCaseCounterClaimPartyRebuttle].dateUpdated, 
[DE_CourtCaseCounterClaimPartyRebuttle].dummyPartyId, 
[DE_CourtCaseCounterClaimPartyRebuttle].courtCaseCounterClaimPartyRebuttleId, 
[DE_CourtCaseCounterClaimPartyRebuttle].courtLegalPosition, 
[DE_CourtCaseCounterClaimPartyRebuttle].courtCaseCounterClaimPartyId, 
[DE_CourtCaseCounterClaimPartyRebuttle].isValidated, 
[DE_CourtCaseCounterClaimPartyRebuttle].updatedUserId
 FROM [DE_CourtCaseCounterClaimPartyRebuttle] INNER JOIN [DE_CourtCaseCounterClaimParty] ON [DE_CourtCaseCounterClaimPartyRebuttle].courtCaseCounterClaimPartyId = [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimPartyId INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttle] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttleDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttleDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttleDocument] ( courtCaseCounterClaimPartyRebuttleId, 
documentId, 
courtCaseCounterClaimPartyRebuttleDocumentId, 
details )
 SELECT [DE_CourtCaseCounterClaimPartyRebuttleDocument].courtCaseCounterClaimPartyRebuttleId, 
[DE_CourtCaseCounterClaimPartyRebuttleDocument].documentId, 
[DE_CourtCaseCounterClaimPartyRebuttleDocument].courtCaseCounterClaimPartyRebuttleDocumentId, 
[DE_CourtCaseCounterClaimPartyRebuttleDocument].details
 FROM [DE_CourtCaseCounterClaimPartyRebuttleDocument] INNER JOIN [DE_CourtCaseCounterClaimPartyRebuttle] ON [DE_CourtCaseCounterClaimPartyRebuttleDocument].courtCaseCounterClaimPartyRebuttleId = [DE_CourtCaseCounterClaimPartyRebuttle].courtCaseCounterClaimPartyRebuttleId INNER JOIN [DE_CourtCaseCounterClaimParty] ON [DE_CourtCaseCounterClaimPartyRebuttle].courtCaseCounterClaimPartyId = [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimPartyId INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyRebuttleDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyDocument] ( courtCaseCounterClaimPartyDocumentId, 
details, 
documentId, 
courtCaseCounterClaimPartyId )
 SELECT [DE_CourtCaseCounterClaimPartyDocument].courtCaseCounterClaimPartyDocumentId, 
[DE_CourtCaseCounterClaimPartyDocument].details, 
[DE_CourtCaseCounterClaimPartyDocument].documentId, 
[DE_CourtCaseCounterClaimPartyDocument].courtCaseCounterClaimPartyId
 FROM [DE_CourtCaseCounterClaimPartyDocument] INNER JOIN [DE_CourtCaseCounterClaimParty] ON [DE_CourtCaseCounterClaimPartyDocument].courtCaseCounterClaimPartyId = [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimPartyId INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimParty].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimPartyDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearing] ( courtCaseCounterClaimId, 
courtCaseCounterClaimHearingId, 
hearingDate )
 SELECT [DE_CourtCaseCounterClaimHearing].courtCaseCounterClaimId, 
[DE_CourtCaseCounterClaimHearing].courtCaseCounterClaimHearingId, 
[DE_CourtCaseCounterClaimHearing].hearingDate
 FROM [DE_CourtCaseCounterClaimHearing] INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimHearing].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearingQuestion] ( partyRoleId, 
partyID, 
questionOrder, 
courtCaseCounterClaimHearingQuestionId, 
question, 
answer, 
representativePartyID, 
courtCaseCounterClaimHearingId )
 SELECT [DE_CourtCaseCounterClaimHearingQuestion].partyRoleId, 
[DE_CourtCaseCounterClaimHearingQuestion].partyInstanceId, 
[DE_CourtCaseCounterClaimHearingQuestion].questionOrder, 
[DE_CourtCaseCounterClaimHearingQuestion].courtCaseCounterClaimHearingQuestionId, 
[DE_CourtCaseCounterClaimHearingQuestion].question, 
[DE_CourtCaseCounterClaimHearingQuestion].answer, 
[DE_CourtCaseCounterClaimHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseCounterClaimHearingQuestion].courtCaseCounterClaimHearingId
 FROM [DE_CourtCaseCounterClaimHearingQuestion] INNER JOIN [DE_CourtCaseCounterClaimHearing] ON [DE_CourtCaseCounterClaimHearingQuestion].courtCaseCounterClaimHearingId = [DE_CourtCaseCounterClaimHearing].courtCaseCounterClaimHearingId INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimHearing].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearing] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearing]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearing] ( hearingDate, 
courtCaseCounterClaimJudgeHearingId, 
judgeComment, 
courtCaseCounterClaimId )
 SELECT [DE_CourtCaseCounterClaimJudgeHearing].hearingDate, 
[DE_CourtCaseCounterClaimJudgeHearing].courtCaseCounterClaimJudgeHearingId, 
[DE_CourtCaseCounterClaimJudgeHearing].judgeComment, 
[DE_CourtCaseCounterClaimJudgeHearing].courtCaseCounterClaimId
 FROM [DE_CourtCaseCounterClaimJudgeHearing] INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimJudgeHearing].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearing] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearingQuestion] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearingQuestion]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearingQuestion] ( partyRoleId, 
courtCaseCounterClaimJudgeHearingId, 
question, 
representativePartyID, 
partyID, 
questionOrder, 
courtCaseCounterClaimJudgeHearingQuestionId )
 SELECT [DE_CourtCaseCounterClaimJudgeHearingQuestion].partyRoleId, 
[DE_CourtCaseCounterClaimJudgeHearingQuestion].courtCaseCounterClaimJudgeHearingId, 
[DE_CourtCaseCounterClaimJudgeHearingQuestion].question, 
[DE_CourtCaseCounterClaimJudgeHearingQuestion].representativePartyInstanceId, 
[DE_CourtCaseCounterClaimJudgeHearingQuestion].partyInstanceId, 
[DE_CourtCaseCounterClaimJudgeHearingQuestion].questionOrder, 
[DE_CourtCaseCounterClaimJudgeHearingQuestion].courtCaseCounterClaimJudgeHearingQuestionId
 FROM [DE_CourtCaseCounterClaimJudgeHearingQuestion] INNER JOIN [DE_CourtCaseCounterClaimJudgeHearing] ON [DE_CourtCaseCounterClaimJudgeHearingQuestion].courtCaseCounterClaimJudgeHearingId = [DE_CourtCaseCounterClaimJudgeHearing].courtCaseCounterClaimJudgeHearingId INNER JOIN [DE_CourtCaseCounterClaim] ON [DE_CourtCaseCounterClaimJudgeHearing].courtCaseCounterClaimId = [DE_CourtCaseCounterClaim].courtCaseCounterClaimId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseCounterClaim].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCounterClaimJudgeHearingQuestion] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDocument] ( courtCaseDocumentId, 
documentId, 
isNoteOrAttachment, CourtCaseID )
 SELECT [DE_CourtCaseDocument].courtCaseDocumentId, 
[DE_CourtCaseDocument].documentId, 
[DE_CourtCaseDocument].isNoteOrAttachment, CourtCaseInstanceID
 FROM [DE_CourtCaseDocument] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDocument].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseJudgeReport] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseJudgeReport]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseJudgeReport] ( isValidated, 
signedCopyDocumentId, 
dateOfJudgment, 
dateAttached, 
hearingDate, 
updatedUserId, 
isDatesObeyed, 
reportDate, 
courtCaseJudgeReportId, 
judgeUserId, 
dateUpdated, 
comment, CourtCaseID )
 SELECT [DE_CourtCaseJudgeReport].isValidated, 
[DE_CourtCaseJudgeReport].signedCopyDocumentId, 
[DE_CourtCaseJudgeReport].dateOfJudgment, 
[DE_CourtCaseJudgeReport].dateAttached, 
[DE_CourtCaseJudgeReport].hearingDate, 
[DE_CourtCaseJudgeReport].updatedUserId, 
[DE_CourtCaseJudgeReport].isDatesObeyed, 
[DE_CourtCaseJudgeReport].reportDate, 
[DE_CourtCaseJudgeReport].courtCaseJudgeReportId, 
[DE_CourtCaseJudgeReport].judgeUserId, 
[DE_CourtCaseJudgeReport].dateUpdated, 
[DE_CourtCaseJudgeReport].comment, CourtCaseInstanceID
 FROM [DE_CourtCaseJudgeReport] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseJudgeReport].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseJudgeReport] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCase] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCase]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCase] ( courtCaseFastTrackCaseId, 
courtGroupId, 
dateFiled, 
fastTrackCourtCaseID, 
WFStateId, 
caseNumber, CourtCaseID )
 SELECT [DE_CourtCaseFastTrackCase].courtCaseFastTrackCaseId, 
[DE_CourtCaseFastTrackCase].courtGroupId, 
[DE_CourtCaseFastTrackCase].dateFiled, 
[DE_CourtCaseFastTrackCase].fastTrackCourtCaseInstanceId, 
[DE_CourtCaseFastTrackCase].WFStateId, 
[DE_CourtCaseFastTrackCase].caseNumber, CourtCaseInstanceID
 FROM [DE_CourtCaseFastTrackCase] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseFastTrackCase].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCase] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCaseDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCaseDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCaseDocument] ( documentId, 
courtCaseFastTrackCaseDocumentId, 
courtCaseFastTrackCaseId )
 SELECT [DE_CourtCaseFastTrackCaseDocument].documentId, 
[DE_CourtCaseFastTrackCaseDocument].courtCaseFastTrackCaseDocumentId, 
[DE_CourtCaseFastTrackCaseDocument].courtCaseFastTrackCaseId
 FROM [DE_CourtCaseFastTrackCaseDocument] INNER JOIN [DE_CourtCaseFastTrackCase] ON [DE_CourtCaseFastTrackCaseDocument].courtCaseFastTrackCaseId = [DE_CourtCaseFastTrackCase].courtCaseFastTrackCaseId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseFastTrackCase].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseFastTrackCaseDocument] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseParty] ( isPartyActive, 
courtCasePartyId, 
partyID, CourtCaseID )
 SELECT [DE_CourtCaseParty].isPartyActive, 
[DE_CourtCaseParty].courtCasePartyId, 
[DE_CourtCaseParty].partyInstanceId, CourtCaseInstanceID
 FROM [DE_CourtCaseParty] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRelationship] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRelationship]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRelationship] ( otherRelationshipType, 
courtCasePartyRelationshipId, 
relativeRoleId, 
relativePartyID, 
details, 
courtCasePartyId )
 SELECT [DE_CourtCasePartyRelationship].otherRelationshipType, 
[DE_CourtCasePartyRelationship].courtCasePartyRelationshipId, 
[DE_CourtCasePartyRelationship].relativeRoleId, 
[DE_CourtCasePartyRelationship].relativePartyInstanceId, 
[DE_CourtCasePartyRelationship].details, 
[DE_CourtCasePartyRelationship].courtCasePartyId
 FROM [DE_CourtCasePartyRelationship] INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCasePartyRelationship].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRelationship] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRole] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRole]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRole] ( isActive, 
courtCasePartyRoleId, 
courtCasePartyId, 
partyRoleId )
 SELECT [DE_CourtCasePartyRole].isActive, 
[DE_CourtCasePartyRole].courtCasePartyRoleId, 
[DE_CourtCasePartyRole].courtCasePartyId, 
[DE_CourtCasePartyRole].partyRoleId
 FROM [DE_CourtCasePartyRole] INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCasePartyRole].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyRole] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyLegalRepresentative] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyLegalRepresentative]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyLegalRepresentative] ( courtCasePartyLegalRepresentativeId, 
representativePartyID, 
courtCasePartyId, 
representativeTypeId, 
isActive )
 SELECT [DE_CourtCasePartyLegalRepresentative].courtCasePartyLegalRepresentativeId, 
[DE_CourtCasePartyLegalRepresentative].representativePartyInstanceId, 
[DE_CourtCasePartyLegalRepresentative].courtCasePartyId, 
[DE_CourtCasePartyLegalRepresentative].representativeTypeId, 
[DE_CourtCasePartyLegalRepresentative].isActive
 FROM [DE_CourtCasePartyLegalRepresentative] INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCasePartyLegalRepresentative].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCasePartyLegalRepresentative] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetails] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetails]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetails] ( courtCaseDecisionDetailsId, 
isSubmitted, 
acceptedId, 
relatedDecisionCategoryId, 
courtCasePartyId )
 SELECT [DE_CourtCaseDecisionDetails].courtCaseDecisionDetailsId, 
[DE_CourtCaseDecisionDetails].isSubmitted, 
[DE_CourtCaseDecisionDetails].acceptedId, 
[DE_CourtCaseDecisionDetails].relatedDecisionCategoryId, 
[DE_CourtCaseDecisionDetails].courtCasePartyId
 FROM [DE_CourtCaseDecisionDetails] INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCaseDecisionDetails].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetails] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetailsSubject] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetailsSubject]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetailsSubject] ( isMainSubject, 
year, 
decisionTypeId, 
diAmount, 
penaltyForcingExecution, 
advocateFees, 
daysInCustody, 
courtProcedureCost, 
startDate, 
day, 
daysInProvisionalDetention, 
restitution, 
damages, 
endDate, 
month, 
fineAmount, 
wonLostId, 
bailAmount, 
courtCaseDecisionDetailsSubjectId, 
courtFees, 
courtCaseDecisionDetailsId, 
decisionDetails, 
shouldBeReleased, 
caseDecisionResolutionId, 
isLifeImprisonment, 
bailConditions, 
rcsEstablishmentGroupId )
 SELECT [DE_CourtCaseDecisionDetailsSubject].isMainSubject, 
[DE_CourtCaseDecisionDetailsSubject].year, 
[DE_CourtCaseDecisionDetailsSubject].decisionTypeId, 
[DE_CourtCaseDecisionDetailsSubject].diAmount, 
[DE_CourtCaseDecisionDetailsSubject].penaltyForcingExecution, 
[DE_CourtCaseDecisionDetailsSubject].advocateFees, 
[DE_CourtCaseDecisionDetailsSubject].daysInCustody, 
[DE_CourtCaseDecisionDetailsSubject].courtProcedureCost, 
[DE_CourtCaseDecisionDetailsSubject].startDate, 
[DE_CourtCaseDecisionDetailsSubject].day, 
[DE_CourtCaseDecisionDetailsSubject].daysInProvisionalDetention, 
[DE_CourtCaseDecisionDetailsSubject].restitution, 
[DE_CourtCaseDecisionDetailsSubject].damages, 
[DE_CourtCaseDecisionDetailsSubject].endDate, 
[DE_CourtCaseDecisionDetailsSubject].month, 
[DE_CourtCaseDecisionDetailsSubject].fineAmount, 
[DE_CourtCaseDecisionDetailsSubject].wonLostId, 
[DE_CourtCaseDecisionDetailsSubject].bailAmount, 
[DE_CourtCaseDecisionDetailsSubject].courtCaseDecisionDetailsSubjectId, 
[DE_CourtCaseDecisionDetailsSubject].courtFees, 
[DE_CourtCaseDecisionDetailsSubject].courtCaseDecisionDetailsId, 
[DE_CourtCaseDecisionDetailsSubject].decisionDetails, 
[DE_CourtCaseDecisionDetailsSubject].shouldBeReleased, 
[DE_CourtCaseDecisionDetailsSubject].caseDecisionResolutionId, 
[DE_CourtCaseDecisionDetailsSubject].isLifeImprisonment, 
[DE_CourtCaseDecisionDetailsSubject].bailConditions, 
[DE_CourtCaseDecisionDetailsSubject].rcsEstablishmentGroupId
 FROM [DE_CourtCaseDecisionDetailsSubject] INNER JOIN [DE_CourtCaseDecisionDetails] ON [DE_CourtCaseDecisionDetailsSubject].courtCaseDecisionDetailsId = [DE_CourtCaseDecisionDetails].courtCaseDecisionDetailsId INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCaseDecisionDetails].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDecisionDetailsSubject] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCrimeType] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCrimeType]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCrimeType] ( details, 
subCategoryId, 
courtCaseCrimeTypeId, 
categoryId, 
offenderTypeId, 
typeId, 
courtCasePartyId, 
articleId )
 SELECT [DE_CourtCaseCrimeType].details, 
[DE_CourtCaseCrimeType].subCategoryId, 
[DE_CourtCaseCrimeType].courtCaseCrimeTypeId, 
[DE_CourtCaseCrimeType].categoryId, 
[DE_CourtCaseCrimeType].offenderTypeId, 
[DE_CourtCaseCrimeType].typeId, 
[DE_CourtCaseCrimeType].courtCasePartyId, 
[DE_CourtCaseCrimeType].articleId
 FROM [DE_CourtCaseCrimeType] INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCaseCrimeType].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseCrimeType] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOffence] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOffence]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOffence] ( address, 
courtCaseCrimeTypeId, 
districtId, 
countryId, 
cellId, 
villageId, 
courtCaseOffenceId, 
provinceId, 
sectorId, 
date )
 SELECT [DE_CourtCaseOffence].address, 
[DE_CourtCaseOffence].courtCaseCrimeTypeId, 
[DE_CourtCaseOffence].districtId, 
[DE_CourtCaseOffence].countryId, 
[DE_CourtCaseOffence].cellId, 
[DE_CourtCaseOffence].villageId, 
[DE_CourtCaseOffence].courtCaseOffenceId, 
[DE_CourtCaseOffence].provinceId, 
[DE_CourtCaseOffence].sectorId, 
[DE_CourtCaseOffence].date
 FROM [DE_CourtCaseOffence] INNER JOIN [DE_CourtCaseCrimeType] ON [DE_CourtCaseOffence].courtCaseCrimeTypeId = [DE_CourtCaseCrimeType].courtCaseCrimeTypeId INNER JOIN [DE_CourtCaseParty] ON [DE_CourtCaseCrimeType].courtCasePartyId = [DE_CourtCaseParty].courtCasePartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseOffence] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovable] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovable]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovable] ( immovables, 
isValidated, 
orderNo, 
courtCaseSeizedMovableId, 
debtAmount, 
documentId, 
attachedDate, 
movables, 
registrarUserId, 
orderDate, 
dateUpdated, CourtCaseID )
 SELECT [DE_CourtCaseSeizedMovable].immovables, 
[DE_CourtCaseSeizedMovable].isValidated, 
[DE_CourtCaseSeizedMovable].orderNo, 
[DE_CourtCaseSeizedMovable].courtCaseSeizedMovableId, 
[DE_CourtCaseSeizedMovable].debtAmount, 
[DE_CourtCaseSeizedMovable].documentId, 
[DE_CourtCaseSeizedMovable].attachedDate, 
[DE_CourtCaseSeizedMovable].movables, 
[DE_CourtCaseSeizedMovable].registrarUserId, 
[DE_CourtCaseSeizedMovable].orderDate, 
[DE_CourtCaseSeizedMovable].dateUpdated, CourtCaseInstanceID
 FROM [DE_CourtCaseSeizedMovable] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSeizedMovable].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovable] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovableParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovableParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovableParty] ( courtCaseSeizedMovableId, 
courtCaseSeizedMovablePartyId, 
partyRoleId, 
partyID, 
isDebtor )
 SELECT [DE_CourtCaseSeizedMovableParty].courtCaseSeizedMovableId, 
[DE_CourtCaseSeizedMovableParty].courtCaseSeizedMovablePartyId, 
[DE_CourtCaseSeizedMovableParty].partyRoleId, 
[DE_CourtCaseSeizedMovableParty].partyInstanceId, 
[DE_CourtCaseSeizedMovableParty].isDebtor
 FROM [DE_CourtCaseSeizedMovableParty] INNER JOIN [DE_CourtCaseSeizedMovable] ON [DE_CourtCaseSeizedMovableParty].courtCaseSeizedMovableId = [DE_CourtCaseSeizedMovable].courtCaseSeizedMovableId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseSeizedMovable].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseSeizedMovableParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroduction] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroduction]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroduction] ( objectOfLitigation, 
dateUpdated, 
closingDate, 
registrarUserId, 
documentId, 
courtCaseReintroductionId, 
isValidated, 
reintroductionDate, 
reintroducedCaseNo, CourtCaseID )
 SELECT [DE_CourtCaseReintroduction].objectOfLitigation, 
[DE_CourtCaseReintroduction].dateUpdated, 
[DE_CourtCaseReintroduction].closingDate, 
[DE_CourtCaseReintroduction].registrarUserId, 
[DE_CourtCaseReintroduction].documentId, 
[DE_CourtCaseReintroduction].courtCaseReintroductionId, 
[DE_CourtCaseReintroduction].isValidated, 
[DE_CourtCaseReintroduction].reintroductionDate, 
[DE_CourtCaseReintroduction].reintroducedCaseNo, CourtCaseInstanceID
 FROM [DE_CourtCaseReintroduction] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseReintroduction].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroduction] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroductionParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroductionParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroductionParty] ( isIntervening, 
partyRoleId, 
partyID, 
courtCaseReintroductionPartyId, 
courtCaseReintroductionId )
 SELECT [DE_CourtCaseReintroductionParty].isIntervening, 
[DE_CourtCaseReintroductionParty].partyRoleId, 
[DE_CourtCaseReintroductionParty].partyInstanceId, 
[DE_CourtCaseReintroductionParty].courtCaseReintroductionPartyId, 
[DE_CourtCaseReintroductionParty].courtCaseReintroductionId
 FROM [DE_CourtCaseReintroductionParty] INNER JOIN [DE_CourtCaseReintroduction] ON [DE_CourtCaseReintroductionParty].courtCaseReintroductionId = [DE_CourtCaseReintroduction].courtCaseReintroductionId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseReintroduction].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseReintroductionParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyParty] ( courtCaseDummyPartyId, 
partyID, 
dummyPartyId, CourtCaseID )
 SELECT [DE_CourtCaseDummyParty].courtCaseDummyPartyId, 
[DE_CourtCaseDummyParty].partyInstanceId, 
[DE_CourtCaseDummyParty].dummyPartyId, CourtCaseInstanceID
 FROM [DE_CourtCaseDummyParty] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDummyParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyPartyLegalRepresentative] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyPartyLegalRepresentative]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyPartyLegalRepresentative] ( representativeDummyPartyId, 
courtCaseDummyPartyLegalRepresentativeId, 
courtCaseDummyPartyId, 
representativeTypeId )
 SELECT [DE_CourtCaseDummyPartyLegalRepresentative].representativeDummyPartyId, 
[DE_CourtCaseDummyPartyLegalRepresentative].courtCaseDummyPartyLegalRepresentativeId, 
[DE_CourtCaseDummyPartyLegalRepresentative].courtCaseDummyPartyId, 
[DE_CourtCaseDummyPartyLegalRepresentative].representativeTypeId
 FROM [DE_CourtCaseDummyPartyLegalRepresentative] INNER JOIN [DE_CourtCaseDummyParty] ON [DE_CourtCaseDummyPartyLegalRepresentative].courtCaseDummyPartyId = [DE_CourtCaseDummyParty].courtCaseDummyPartyId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseDummyParty].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseDummyPartyLegalRepresentative] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIntervention] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIntervention]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIntervention] ( isValidated, 
officialTellingToSign, 
post, 
phone, 
date, 
interventionTypeId, 
courtCaseInterventionId, 
documentId, 
interventionDate, 
email, 
hearingDate, 
registrarUserId, 
objectOfLitigation, 
dateUpdated, CourtCaseID )
 SELECT [DE_CourtCaseIntervention].isValidated, 
[DE_CourtCaseIntervention].officialTellingToSign, 
[DE_CourtCaseIntervention].post, 
[DE_CourtCaseIntervention].phone, 
[DE_CourtCaseIntervention].date, 
[DE_CourtCaseIntervention].interventionTypeId, 
[DE_CourtCaseIntervention].courtCaseInterventionId, 
[DE_CourtCaseIntervention].documentId, 
[DE_CourtCaseIntervention].interventionDate, 
[DE_CourtCaseIntervention].email, 
[DE_CourtCaseIntervention].hearingDate, 
[DE_CourtCaseIntervention].registrarUserId, 
[DE_CourtCaseIntervention].objectOfLitigation, 
[DE_CourtCaseIntervention].dateUpdated, CourtCaseInstanceID
 FROM [DE_CourtCaseIntervention] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIntervention].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseIntervention] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseInterventionParty] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseInterventionParty]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseInterventionParty] ( isIntervening, 
courtCaseInterventionId, 
courtCaseInterventionPartyId, 
partyRoleId, 
partyID )
 SELECT [DE_CourtCaseInterventionParty].isIntervening, 
[DE_CourtCaseInterventionParty].courtCaseInterventionId, 
[DE_CourtCaseInterventionParty].courtCaseInterventionPartyId, 
[DE_CourtCaseInterventionParty].partyRoleId, 
[DE_CourtCaseInterventionParty].partyInstanceId
 FROM [DE_CourtCaseInterventionParty] INNER JOIN [DE_CourtCaseIntervention] ON [DE_CourtCaseInterventionParty].courtCaseInterventionId = [DE_CourtCaseIntervention].courtCaseInterventionId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseIntervention].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseInterventionParty] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedure] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedure]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedure] ( courtCaseExtraordinaryProcedureId, 
dateFiled, 
extraordinaryProcedureCourtCaseID, 
courtGroupId, 
caseNumber, 
WFStateId, CourtCaseID )
 SELECT [DE_CourtCaseExtraordinaryProcedure].courtCaseExtraordinaryProcedureId, 
[DE_CourtCaseExtraordinaryProcedure].dateFiled, 
[DE_CourtCaseExtraordinaryProcedure].extraordinaryProcedureCourtCaseInstanceId, 
[DE_CourtCaseExtraordinaryProcedure].courtGroupId, 
[DE_CourtCaseExtraordinaryProcedure].caseNumber, 
[DE_CourtCaseExtraordinaryProcedure].WFStateId, CourtCaseInstanceID
 FROM [DE_CourtCaseExtraordinaryProcedure] INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseExtraordinaryProcedure].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedure] OFF 
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedureDocument] ON 
 DELETE FROM [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedureDocument]
 INSERT INTO [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedureDocument] ( documentId, 
courtCaseExtraordinaryProcedureDocumentId, 
courtCaseExtraordinaryProcedureId )
 SELECT [DE_CourtCaseExtraordinaryProcedureDocument].documentId, 
[DE_CourtCaseExtraordinaryProcedureDocument].courtCaseExtraordinaryProcedureDocumentId, 
[DE_CourtCaseExtraordinaryProcedureDocument].courtCaseExtraordinaryProcedureId
 FROM [DE_CourtCaseExtraordinaryProcedureDocument] INNER JOIN [DE_CourtCaseExtraordinaryProcedure] ON [DE_CourtCaseExtraordinaryProcedureDocument].courtCaseExtraordinaryProcedureId = [DE_CourtCaseExtraordinaryProcedure].courtCaseExtraordinaryProcedureId INNER JOIN [DE_CourtCasePublishedItem] ON [DE_CourtCaseExtraordinaryProcedure].courtCaseId = [DE_CourtCasePublishedItem].courtCaseId
 SET IDENTITY_INSERT [DEV-RWA_IECMS-DATA].dbo.[CourtCaseExtraordinaryProcedureDocument] OFF  End 