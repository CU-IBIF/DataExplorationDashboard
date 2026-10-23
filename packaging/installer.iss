; Inno Setup script: wraps the PyInstaller folder (dist\FeatureScope) into FeatureScope-Setup-<version>.exe.
; Per-user install (no administrator rights needed), Start-menu shortcut, optional desktop icon.
;   iscc /DAppVersion=0.1.0 packaging\installer.iss
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6A0F5C1E-7E1B-4C55-9D1B-6C2B7E2F3A10}
AppName=FeatureScope
AppVersion={#AppVersion}
AppPublisher=CU-IBIF
AppPublisherURL=https://github.com/CU-IBIF/DataExplorationDashboard
DefaultDirName={autopf}\FeatureScope
DefaultGroupName=FeatureScope
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=FeatureScope-Setup-{#AppVersion}
SetupIconFile=featurescope.ico
UninstallDisplayIcon={app}\FeatureScope.exe
Compression=lzma2/ultra64
SolidCompression=yes
LZMANumBlockThreads=4
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\FeatureScope\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\FeatureScope"; Filename: "{app}\FeatureScope.exe"
Name: "{group}\Uninstall FeatureScope"; Filename: "{uninstallexe}"
Name: "{autodesktop}\FeatureScope"; Filename: "{app}\FeatureScope.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FeatureScope.exe"; Description: "Launch FeatureScope"; Flags: nowait postinstall skipifsilent
