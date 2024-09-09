"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> tensor<complex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(1.000000e+00,1.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    "func.return"(%0) : (tensor<complex<f32>>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

