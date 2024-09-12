"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x100x100xi8>) -> tensor<?x100x100xcomplex<f64>>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x100x100xi8>):
    %0 = "stablehlo.convert"(%arg1) : (tensor<?x100x100xi8>) -> tensor<?x100x100xcomplex<f64>>
    "func.return"(%0) : (tensor<?x100x100xcomplex<f64>>) -> ()
  }) : () -> ()
}) : () -> ()

