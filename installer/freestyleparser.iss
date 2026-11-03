; Inno Setup: инсталлятор FreestyleParser для Windows
; Собирается CI (choco install innosetup; ISCC installer/freestyleparser.iss)
; Версия передаётся дефайном: ISCC ... /DAppVersion=1.3.3

#ifndef AppVersion
#define AppVersion "0.0.0"
#endif

#define AppName "Freestyle Parser"
#define AppExe "FreestyleParser.exe"

[Setup]
AppId={{2E3334E8-9C08-45FD-A463-3EE14BD82059}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\FreestyleParser
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; не требуем админа, если ставим только в Program Files - всё равно нужен;
; privileged для автозапуска не нужен
OutputDir=dist
OutputBaseFilename=FreestyleParser-setup-{#AppVersion}
SetupIconFile=icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; установка на x64 без сворачивания в Program Files (x86)
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; содержимое onedir-сборки PyInstaller (dist/FreestyleParser/*)
Source: "dist\FreestyleParser\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; ничего пользовательского не удаляем: папки дней (Desktop\FreestyleParser)
; и ~/.FreestyleParser остаются на месте
