kernel void kernel_trunc_withoutDD1(global float* result_trunc, int N) {
  float t1;
  float t2;
  float t3;
  float t4;
  float t5;
  float t6;
  float t7;
  float t8;
  float t9;
  float t10;

  float i = 1.0;
  float n = 0.3 * (float)(N);
  for (i = 1.0; i < n; i = i + 0.01) {
    t1 = trunc(i);
    t2 = trunc(i + 0.1);
    t3 = trunc(i + 0.2);
    t4 = trunc(i + 0.3);
    t5 = trunc(i + 0.4);
    t6 = trunc(i + 0.5);
    t7 = trunc(i + 0.6);
    t8 = trunc(i + 0.7);
    t9 = trunc(i + 0.8);
    t10 = trunc(i + 0.9);
  }
  *result_trunc = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10;
}
