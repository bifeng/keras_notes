Python3 的 f-string 字符串格式化速度超快，用起来！
f'{s} {t}'  -> 78.2 ns
s + '  ' + t  -> 104 ns
' '.join((s, t))  -> 135 ns
'%s %s' % (s, t) -> 188 ns
'{} {}'.format(s, t) -> 283 ns
Template('$s $t').substitute(s=s, t=t)  -> 898 ns
via:Raymond Hettinger