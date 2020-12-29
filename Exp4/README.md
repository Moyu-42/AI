一点建议

第134行的训练轮数 epochs 可以设置的小一点，比如 5，10 结果都差不多，轮数太多训练的太慢

训练一次之后会将模型保存在 'cnn.pt' (line 137)
下次使用时可以注释掉 line 136: cnn.fit(data_loader_train, data_loader_test)
将 line 135: cnn = torch.load('cnn.pt') 取消注释
从而直接加载训练好的模型

对于最后输出图片 可以在终端使用 Ctrl-C 结束图片的显示（当然，直接把终端关掉也是可以的，只不过这不优雅）
