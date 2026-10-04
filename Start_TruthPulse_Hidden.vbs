' Launch TruthPulse completely in the background without any terminal window
Set WshShell = CreateObject("WScript.Shell")
strPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Open browser
WshShell.Run "cmd /c start http://localhost:8501", 0, False

' Run streamlit quietly in background
WshShell.Run "python -m streamlit run """ & strPath & "\ui\app.py"" --server.port 8501 --server.headless true", 0, False
