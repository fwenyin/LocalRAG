ALTER TABLE CommonVariables
MODIFY `ID_short` VARCHAR(255) NOT NULL;

ALTER TABLE CommonVariables
ADD PRIMARY KEY (`ID_short`);

ALTER TABLE Anthro
MODIFY `ID_short` VARCHAR(255) NOT NULL;

ALTER TABLE Anthro 
ADD FOREIGN KEY (`ID_short`) REFERENCES CommonVariables (`ID_short`);

ALTER TABLE BloodValue
MODIFY `ID_short` VARCHAR(255) NOT NULL;

ALTER TABLE BloodValue
ADD FOREIGN KEY (`ID_short`) REFERENCES CommonVariables (`ID_short`);

ALTER TABLE Nutrients
MODIFY `ID_short` VARCHAR(255) NOT NULL;

ALTER TABLE Nutrients
ADD FOREIGN KEY (`ID_short`) REFERENCES CommonVariables (`ID_short`);
