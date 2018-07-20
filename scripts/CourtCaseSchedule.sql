SELECT  CourtCaseID ,        
		DATEDIFF(DAY, '1900-01-01', IssueDate) AS IssueDate ,
		DATEDIFF(DAY, '1900-01-01', DateUpdated) AS DateUpdated ,      
        HearingTypeID ,
        DATEDIFF(DAY, '1900-01-01', HearingDate) AS HearingDate ,
        RegistrarUserID ,
        DocumentID ,
        AttachedDate ,
        IsValidated
        HearingStatusID ,
        CourtCaseScheduleID
FROM    dbo.CourtCaseSchedule;
