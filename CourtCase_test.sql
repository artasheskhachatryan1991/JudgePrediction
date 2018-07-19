;
WITH    cte
          AS ( SELECT   MajorVersion ,
                        MinorVersion ,
                        DATEDIFF(DAY,'1900-01-01', CaseSubmittedDate)CaseSubmittedDate ,
                        DATEDIFF(DAY,'1900-01-01', CaseRegisteredDate) CaseRegisteredDate,
                        CasePriorityID ,
                        CourtID ,
                        CourtPresidentUserID ,
                        ChiefRegistrarUserID ,
                        AssignedRegistrarUserID ,
                        AssignedJudgeUserID ,
                        OwnerUserID ,
                        UpdatedUserID ,
                        WFStateID ,
                        IsExempted ,
                        PaymentBankID ,
                        WFActionID ,
                        InitiatedFromAbunzi ,
                        SolvedFromAbunzi ,
                        CreatedUserID ,
                        DATEDIFF(DAY,'1900-01-01',DateCreated) DateCreated,
                        CaseCode ,
                        NotRegisteredCaseCode ,
                        PreviousCourtCaseID ,                        
                        DATEDIFF(DAY,'1900-01-01',DecisionPronouncementDate) DecisionPronouncementDate ,
                        ReceiptDocumentID ,
                        DATEDIFF(DAY,'1900-01-01', AttachedDate) AttachedDate ,
                        AppealedCourtCaseID ,
                        ProsecutionCaseID ,
                        InstanceLevelID ,
                        DateCreatedYearID ,
                        DecisionPronouncementDateYearID ,
                        SpecialCaseID ,
                        CommittedByMinor ,
                        GenderBasedViolence ,
                        LitigationCaseID ,
                        IsPublicCase ,
                        ExtraOrdinaryProcedureID ,
                        CategoryID ,
                        SubCategoryID ,
                        PublicOwnerUserId ,
                        FillingFee ,
                        ColorID ,
                        CountOfJudgmentPages ,
                        CourtCaseID ,
                        IsDetentionCase ,
                        CaseInitialID ,
                        HasDetails ,
                        HasPassedCaseNumberAllocated ,
                        ExecutionCaseApprovedUserID ,
                        CaseRejectionID
               FROM     dbo.CourtCase
               WHERE    DateCreated > '2018-01-01'
             )

			 , cte1 AS (
    SELECT  * ,
            ( SELECT    DATEDIFF(DAY, '1900-01-01', MAX(ccwf.ActionDate))
              FROM      dbo.CourtCaseWFAction ccwf
              WHERE     ResultingStateID IN ( 120)
                        AND ccwf.CourtCaseID = cte.CourtCaseID
              GROUP BY  ccwf.CourtCaseID
            ) AS DecisionEndDate
			 ,
            ( SELECT    DATEDIFF(DAY, '1900-01-01', MAX(ccwf.ActionDate))
              FROM      dbo.CourtCaseWFAction ccwf
              WHERE     ResultingStateID IN ( 81)
                        AND ccwf.CourtCaseID = cte.CourtCaseID
              GROUP BY  ccwf.CourtCaseID
            ) AS DecisionStartDate
    FROM    cte
	)

	SELECT MajorVersion ,
                        MinorVersion ,
                        CaseSubmittedDate ,
                        CaseRegisteredDate,
                        CasePriorityID ,
                        CourtID ,
                        CourtPresidentUserID ,
                        ChiefRegistrarUserID ,
                        AssignedRegistrarUserID ,
                        AssignedJudgeUserID ,
                        OwnerUserID ,
                        UpdatedUserID ,
                        WFStateID ,
                        IsExempted ,
                        PaymentBankID ,
                        WFActionID ,
                        InitiatedFromAbunzi ,
                        SolvedFromAbunzi ,
                        CreatedUserID ,
                        DateCreated,
                        CaseCode ,
                        NotRegisteredCaseCode ,
                        PreviousCourtCaseID ,                        
                        DecisionPronouncementDate ,
                        ReceiptDocumentID ,
                        AttachedDate ,
                        AppealedCourtCaseID ,
                        ProsecutionCaseID ,
                        InstanceLevelID ,
                        DateCreatedYearID ,
                        DecisionPronouncementDateYearID ,
                        SpecialCaseID ,
                        CommittedByMinor ,
                        GenderBasedViolence ,
                        LitigationCaseID ,
                        IsPublicCase ,
                        ExtraOrdinaryProcedureID ,
                        CategoryID ,
                        SubCategoryID ,
                        PublicOwnerUserId ,
                        FillingFee ,
                        ColorID ,
                        CountOfJudgmentPages ,
                        CourtCaseID ,
                        IsDetentionCase ,
                        CaseInitialID ,
                        HasDetails ,
                        HasPassedCaseNumberAllocated ,
                        ExecutionCaseApprovedUserID ,
                        CaseRejectionID, 
						DecisionEndDate - DecisionStartDate AS DecisionDuration
	FROM cte1
	WHERE DecisionEndDate - DecisionStartDate IS NOT NULL
	AND DATEADD(DAY, cte1.DecisionEndDate, '1900-01-01') >= '2018-06-15'
	

--SELECT dbo.Concat(COLUMN_NAME, ',')
--FROM INFORMATION_SCHEMA.COLUMNS
--WHERE TABLE_NAME = 'CourtCaseWFAction' AND DATA_TYPE NOT IN ('varchar', 'nvarchar')
--GROUP BY TABLE_NAME






