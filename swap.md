通过此种方式进行swap 的扩展，首先要计算出block的数目。具体为根据需要扩展的swapfile的大小，以M为单位。block=swap分区大小*1M, 例如，需要扩展8G的swapfile，则：block=8192*1M=8G.
然后做如下步骤：
```
# dd if=/dev/zero of=/mnt/swapfile bs=1M count=8192
2. 创建SWAP文件
# mkswap /mnt/swapfile
3. 激活SWAP文件
# swapon /mnt/swapfile
4. 查看SWAP信息是否正确
# swapon -s
5. 添加到fstab文件中让系统引导时自动启动
# echo "/mnt/swapfile swap swap defaults 0 0" >> /etc/fstab
6. 用命令free检查2G交换分区生效
# free -m
# grep SwapTotal  /proc/meminfo
7. 释放SWAP文件
# swapoff /mnt/swapfile
8. 删除SWAP文件
# rm -fr /mnt/swapfile
```