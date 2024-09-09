"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<9xui32>) -> tensor<3x3xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<9xui32>):
    %0 = "stablehlo.reshape"(%arg0) : (tensor<9xui32>) -> tensor<3x3xui32>
    %1 = "stablehlo.transpose"(%0) <{permutation = array<i64: 1, 0>}> : (tensor<3x3xui32>) -> tensor<3x3xui32>
    "func.return"(%1) : (tensor<3x3xui32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

