import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
y = np.sin(x)



with open("test5_processed.csv", "r") as f:
    # a = f.readline()
    # print(a)
    # xd = f.readline().split(",")
    # yd = f.readline().split(",")
    # xdd = f.readline().split(",")
    # ydd = f.readline().split(",")

    x = np.array([float(x) for x in f.readline().split(",") if x != "\n"])
    # print("x is: ")
    # print(x)
    # pause = input("pausing")
    y = np.array([float(x) for x in f.readline().split(",") if x != "\n"])
    xd = np.array([float(x) for x in f.readline().split(",") if x != "\n"])
    yd = np.array([float(x) for x in f.readline().split(",") if x != "\n"])
    xdd = np.array([float(x) for x in f.readline().split(",") if x != "\n"])
    ydd = np.array([float(x) for x in f.readline().split(",") if x != "\n"])
    # y = np.array(f.readline().split(","))
    # xd = np.array(f.readline().split(","))
    # yd = np.array(f.readline().split(","))
    # xdd = np.array(f.readline().split(","))
    # ydd = np.array(f.readline().split(","))



T = len(x)
t = np.arange(T)
t_norming = np.arange(0,T,float(T)/1000)
t_long = np.arange(1000)

# tnew = t / 50.
# print("t is: ")
# print(t)
# print("tnew is: ")
# print(tnew)

k_val = 5
xb=0.
xe=T
# tck = interpolate.splrep(x, y, s=0, k=3)
tck = interpolate.splrep(t, x, k=k_val, xb=0, xe=xe)
# tck_long = interpolate.splrep(t, x, k=k_val, xb=0, xe=xe)
x_new = interpolate.splev(t, tck, der=0)
x_new_long = interpolate.splev(t_norming, tck, der=0) # THIS IS THE KEY RIGHT HERE!!!
print("x_new_long is: ")
print(np.shape(x_new_long))
print(x_new_long)


tck_xd_ = interpolate.splrep(t, x, k=k_val, xb=0, xe=xe)
xd_new_ = interpolate.splev(t, tck, der=1)

tck_xdd_ = interpolate.splrep(t, x, k=k_val, xb=0, xe=xe)
xdd_new_ = interpolate.splev(t, tck, der=2)

tck_y = interpolate.splrep(t, y, k=k_val, xb=xb, xe=xe)
y_new = interpolate.splev(t, tck_y, der=0)

tck_xd = interpolate.splrep(t, xd, k=k_val, xb=xb, xe=xe)
xd_new = interpolate.splev(t, tck_xd, der=0)

tck_yd = interpolate.splrep(t, yd, k=k_val, xb=xb, xe=xe)
yd_new = interpolate.splev(t, tck_yd, der=0)

tck_xdd = interpolate.splrep(t, xdd, k=k_val, xb=xb, xe=xe)
xdd_new = interpolate.splev(t, tck_xdd, der=0)

tck_ydd = interpolate.splrep(t, ydd, k=k_val, xb=xb, xe=xe)
ydd_new = interpolate.splev(t, tck_ydd, der=0)

plt.figure()

plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t, x_new, "b", t_long, x_new_long, 'r')
# plt.plot(t, x_new, 'b', t_long, x_new_long, 'r')
plt.title('xlong vs t')
plt.show()


plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t, x_new, 'b', t, xd_new_, 'r')
plt.title('x vs t')
plt.show()

plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t[0:100], xd_new[0:100], 'b', t[0:100], xd_new_[0:100], 'r')
plt.title('xd vs t')
plt.show()

plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t[0:200], xdd_new[0:200], 'b', t[0:200], xdd_new_[0:200], 'r')
plt.title('xdd vs t')
plt.show()

plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t, y, 'b', t, y_new, 'r')
plt.title('y vs t')
plt.show()

plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t, yd, 'b', t, yd_new, 'r')
plt.title('yd vs t')
plt.show()

plt.legend(['Linear', 'Cubic Spline'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.plot(t, ydd, 'b', t, ydd_new, 'r')
plt.title('ydd vs t')
plt.show()

# xnew = np.arange(0,2*np.pi,np.pi/50)
# ynew = interpolate.splev(xnew, tck, der=2)

# plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
# plt.plot(x, y, 'x')
# plt.plot(x, y, 'x', xnew, ynew)
# plt.plot(x, y, 'x', xnew, ynew, xnew + .1, np.sin(xnew), x, y, 'b')
# plt.plot()







# # tck = interpolate.splrep(x, y, s=0, k=3)
# tck = interpolate.splrep(x, y, k=5)
# xnew = np.arange(0,2*np.pi,np.pi/50)
# ynew = interpolate.splev(xnew, tck, der=2)

# plt.figure()
# # plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
# # plt.plot(x, y, 'x')
# # plt.plot(x, y, 'x', xnew, ynew)
# plt.plot(x, y, 'x', xnew, ynew, xnew + .1, np.sin(xnew), x, y, 'b')
# # plt.plot()

# plt.legend(['Linear', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
# plt.title('Cubic-spline interpolation')
# plt.show()
