"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> tensor<3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xbf16>}> : () -> tensor<3xbf16>
    "func.return"(%0) : (tensor<3xbf16>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

