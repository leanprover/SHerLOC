"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2xf32>) -> tensor<?x2xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>):
    "func.return"(%arg1) : (tensor<?x2xf32>) -> ()
  }) : () -> ()
}) : () -> ()

