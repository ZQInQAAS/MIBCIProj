import wx
from windows import MainWindow

app = wx.App()
app.locale = wx.Locale(wx.LANGUAGE_CHINESE_SIMPLIFIED)
win = MainWindow()
win.Show()
app.MainLoop()

