"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf64>
    %4 = "stablehlo.abs"(%2) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %5 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f64>}> : () -> tensor<f64>
    %6 = "stablehlo.broadcast_in_dim"(%5) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %7 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %9 = "stablehlo.constant"() <{value = dense<3.200000e+01> : tensor<f64>}> : () -> tensor<f64>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %11 = "stablehlo.multiply"(%6, %4) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %12 = "stablehlo.subtract"(%11, %8) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %13 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %14 = "stablehlo.broadcast_in_dim"(%13) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %15 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %17 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %19 = "stablehlo.multiply"(%12, %14) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %20 = "stablehlo.subtract"(%19, %16) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %21 = "stablehlo.constant"() <{value = dense<-4.4153416464793395E-18> : tensor<f64>}> : () -> tensor<f64>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %23 = "stablehlo.add"(%20, %22) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %24 = "stablehlo.multiply"(%12, %23) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %25 = "stablehlo.subtract"(%24, %14) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %26 = "stablehlo.constant"() <{value = dense<3.3307945188222384E-17> : tensor<f64>}> : () -> tensor<f64>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %28 = "stablehlo.add"(%25, %27) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %29 = "stablehlo.multiply"(%12, %28) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %30 = "stablehlo.subtract"(%29, %23) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %31 = "stablehlo.constant"() <{value = dense<-2.4312798465479549E-16> : tensor<f64>}> : () -> tensor<f64>
    %32 = "stablehlo.broadcast_in_dim"(%31) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %33 = "stablehlo.add"(%30, %32) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %34 = "stablehlo.multiply"(%12, %33) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %35 = "stablehlo.subtract"(%34, %28) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %36 = "stablehlo.constant"() <{value = dense<1.7153912855551331E-15> : tensor<f64>}> : () -> tensor<f64>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %38 = "stablehlo.add"(%35, %37) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %39 = "stablehlo.multiply"(%12, %38) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %40 = "stablehlo.subtract"(%39, %33) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %41 = "stablehlo.constant"() <{value = dense<-1.1685332877993451E-14> : tensor<f64>}> : () -> tensor<f64>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %43 = "stablehlo.add"(%40, %42) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %44 = "stablehlo.multiply"(%12, %43) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %45 = "stablehlo.subtract"(%44, %38) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %46 = "stablehlo.constant"() <{value = dense<7.6761854986049361E-14> : tensor<f64>}> : () -> tensor<f64>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %48 = "stablehlo.add"(%45, %47) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %49 = "stablehlo.multiply"(%12, %48) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %50 = "stablehlo.subtract"(%49, %43) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %51 = "stablehlo.constant"() <{value = dense<-4.856446783111929E-13> : tensor<f64>}> : () -> tensor<f64>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %53 = "stablehlo.add"(%50, %52) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %54 = "stablehlo.multiply"(%12, %53) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %55 = "stablehlo.subtract"(%54, %48) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %56 = "stablehlo.constant"() <{value = dense<2.9550526631296399E-12> : tensor<f64>}> : () -> tensor<f64>
    %57 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %58 = "stablehlo.add"(%55, %57) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %59 = "stablehlo.multiply"(%12, %58) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %60 = "stablehlo.subtract"(%59, %53) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %61 = "stablehlo.constant"() <{value = dense<-1.7268262914415559E-11> : tensor<f64>}> : () -> tensor<f64>
    %62 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %63 = "stablehlo.add"(%60, %62) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %64 = "stablehlo.multiply"(%12, %63) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %65 = "stablehlo.subtract"(%64, %58) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %66 = "stablehlo.constant"() <{value = dense<9.6758090353732369E-11> : tensor<f64>}> : () -> tensor<f64>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %68 = "stablehlo.add"(%65, %67) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %69 = "stablehlo.multiply"(%12, %68) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %70 = "stablehlo.subtract"(%69, %63) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %71 = "stablehlo.constant"() <{value = dense<-5.1897956016352627E-10> : tensor<f64>}> : () -> tensor<f64>
    %72 = "stablehlo.broadcast_in_dim"(%71) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %73 = "stablehlo.add"(%70, %72) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %74 = "stablehlo.multiply"(%12, %73) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %75 = "stablehlo.subtract"(%74, %68) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %76 = "stablehlo.constant"() <{value = dense<2.6598237246823866E-9> : tensor<f64>}> : () -> tensor<f64>
    %77 = "stablehlo.broadcast_in_dim"(%76) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %78 = "stablehlo.add"(%75, %77) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %79 = "stablehlo.multiply"(%12, %78) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %80 = "stablehlo.subtract"(%79, %73) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %81 = "stablehlo.constant"() <{value = dense<-1.300025009986248E-8> : tensor<f64>}> : () -> tensor<f64>
    %82 = "stablehlo.broadcast_in_dim"(%81) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %83 = "stablehlo.add"(%80, %82) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %84 = "stablehlo.multiply"(%12, %83) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %85 = "stablehlo.subtract"(%84, %78) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %86 = "stablehlo.constant"() <{value = dense<6.0469950225419186E-8> : tensor<f64>}> : () -> tensor<f64>
    %87 = "stablehlo.broadcast_in_dim"(%86) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %88 = "stablehlo.add"(%85, %87) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %89 = "stablehlo.multiply"(%12, %88) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %90 = "stablehlo.subtract"(%89, %83) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %91 = "stablehlo.constant"() <{value = dense<-2.6707938539406119E-7> : tensor<f64>}> : () -> tensor<f64>
    %92 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %93 = "stablehlo.add"(%90, %92) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %94 = "stablehlo.multiply"(%12, %93) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %95 = "stablehlo.subtract"(%94, %88) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %96 = "stablehlo.constant"() <{value = dense<1.1173875391201037E-6> : tensor<f64>}> : () -> tensor<f64>
    %97 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %98 = "stablehlo.add"(%95, %97) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %99 = "stablehlo.multiply"(%12, %98) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %100 = "stablehlo.subtract"(%99, %93) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %101 = "stablehlo.constant"() <{value = dense<-4.4167383584587505E-6> : tensor<f64>}> : () -> tensor<f64>
    %102 = "stablehlo.broadcast_in_dim"(%101) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %103 = "stablehlo.add"(%100, %102) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %104 = "stablehlo.multiply"(%12, %103) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %105 = "stablehlo.subtract"(%104, %98) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %106 = "stablehlo.constant"() <{value = dense<1.6448448070728896E-5> : tensor<f64>}> : () -> tensor<f64>
    %107 = "stablehlo.broadcast_in_dim"(%106) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %108 = "stablehlo.add"(%105, %107) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %109 = "stablehlo.multiply"(%12, %108) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %110 = "stablehlo.subtract"(%109, %103) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %111 = "stablehlo.constant"() <{value = dense<-5.754195010082104E-5> : tensor<f64>}> : () -> tensor<f64>
    %112 = "stablehlo.broadcast_in_dim"(%111) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %113 = "stablehlo.add"(%110, %112) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %114 = "stablehlo.multiply"(%12, %113) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %115 = "stablehlo.subtract"(%114, %108) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %116 = "stablehlo.constant"() <{value = dense<1.8850288509584165E-4> : tensor<f64>}> : () -> tensor<f64>
    %117 = "stablehlo.broadcast_in_dim"(%116) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %118 = "stablehlo.add"(%115, %117) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %119 = "stablehlo.multiply"(%12, %118) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %120 = "stablehlo.subtract"(%119, %113) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %121 = "stablehlo.constant"() <{value = dense<-5.7637557453858236E-4> : tensor<f64>}> : () -> tensor<f64>
    %122 = "stablehlo.broadcast_in_dim"(%121) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %123 = "stablehlo.add"(%120, %122) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %124 = "stablehlo.multiply"(%12, %123) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %125 = "stablehlo.subtract"(%124, %118) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %126 = "stablehlo.constant"() <{value = dense<0.0016394756169413357> : tensor<f64>}> : () -> tensor<f64>
    %127 = "stablehlo.broadcast_in_dim"(%126) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %128 = "stablehlo.add"(%125, %127) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %129 = "stablehlo.multiply"(%12, %128) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %130 = "stablehlo.subtract"(%129, %123) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %131 = "stablehlo.constant"() <{value = dense<-0.0043243099950505759> : tensor<f64>}> : () -> tensor<f64>
    %132 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %133 = "stablehlo.add"(%130, %132) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %134 = "stablehlo.multiply"(%12, %133) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %135 = "stablehlo.subtract"(%134, %128) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %136 = "stablehlo.constant"() <{value = dense<0.010546460394594998> : tensor<f64>}> : () -> tensor<f64>
    %137 = "stablehlo.broadcast_in_dim"(%136) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %138 = "stablehlo.add"(%135, %137) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %139 = "stablehlo.multiply"(%12, %138) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %140 = "stablehlo.subtract"(%139, %133) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %141 = "stablehlo.constant"() <{value = dense<-0.023737414805899471> : tensor<f64>}> : () -> tensor<f64>
    %142 = "stablehlo.broadcast_in_dim"(%141) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %143 = "stablehlo.add"(%140, %142) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %144 = "stablehlo.multiply"(%12, %143) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %145 = "stablehlo.subtract"(%144, %138) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %146 = "stablehlo.constant"() <{value = dense<0.049305284239670712> : tensor<f64>}> : () -> tensor<f64>
    %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %148 = "stablehlo.add"(%145, %147) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %149 = "stablehlo.multiply"(%12, %148) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %150 = "stablehlo.subtract"(%149, %143) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %151 = "stablehlo.constant"() <{value = dense<-0.094901097048047639> : tensor<f64>}> : () -> tensor<f64>
    %152 = "stablehlo.broadcast_in_dim"(%151) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %153 = "stablehlo.add"(%150, %152) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %154 = "stablehlo.multiply"(%12, %153) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %155 = "stablehlo.subtract"(%154, %148) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %156 = "stablehlo.constant"() <{value = dense<0.17162090152220877> : tensor<f64>}> : () -> tensor<f64>
    %157 = "stablehlo.broadcast_in_dim"(%156) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %158 = "stablehlo.add"(%155, %157) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %159 = "stablehlo.multiply"(%12, %158) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %160 = "stablehlo.subtract"(%159, %153) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %161 = "stablehlo.constant"() <{value = dense<-0.3046826723431984> : tensor<f64>}> : () -> tensor<f64>
    %162 = "stablehlo.broadcast_in_dim"(%161) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %163 = "stablehlo.add"(%160, %162) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %164 = "stablehlo.multiply"(%12, %163) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %165 = "stablehlo.subtract"(%164, %158) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %166 = "stablehlo.constant"() <{value = dense<0.67679527440947607> : tensor<f64>}> : () -> tensor<f64>
    %167 = "stablehlo.broadcast_in_dim"(%166) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %168 = "stablehlo.add"(%165, %167) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %169 = "stablehlo.subtract"(%168, %158) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %170 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f64>}> : () -> tensor<f64>
    %171 = "stablehlo.broadcast_in_dim"(%170) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %172 = "stablehlo.multiply"(%171, %169) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %173 = "stablehlo.divide"(%10, %4) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %174 = "stablehlo.subtract"(%173, %8) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %175 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %176 = "stablehlo.broadcast_in_dim"(%175) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %177 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %178 = "stablehlo.broadcast_in_dim"(%177) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %179 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %180 = "stablehlo.broadcast_in_dim"(%179) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %181 = "stablehlo.multiply"(%174, %176) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %182 = "stablehlo.subtract"(%181, %178) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %183 = "stablehlo.constant"() <{value = dense<-7.2331804878747538E-18> : tensor<f64>}> : () -> tensor<f64>
    %184 = "stablehlo.broadcast_in_dim"(%183) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %185 = "stablehlo.add"(%182, %184) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %186 = "stablehlo.multiply"(%174, %185) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %187 = "stablehlo.subtract"(%186, %176) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %188 = "stablehlo.constant"() <{value = dense<-4.8305044859441819E-18> : tensor<f64>}> : () -> tensor<f64>
    %189 = "stablehlo.broadcast_in_dim"(%188) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %190 = "stablehlo.add"(%187, %189) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %191 = "stablehlo.multiply"(%174, %190) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %192 = "stablehlo.subtract"(%191, %185) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %193 = "stablehlo.constant"() <{value = dense<4.4656214202967598E-17> : tensor<f64>}> : () -> tensor<f64>
    %194 = "stablehlo.broadcast_in_dim"(%193) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %195 = "stablehlo.add"(%192, %194) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %196 = "stablehlo.multiply"(%174, %195) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %197 = "stablehlo.subtract"(%196, %190) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %198 = "stablehlo.constant"() <{value = dense<3.4612228676974612E-17> : tensor<f64>}> : () -> tensor<f64>
    %199 = "stablehlo.broadcast_in_dim"(%198) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %200 = "stablehlo.add"(%197, %199) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %201 = "stablehlo.multiply"(%174, %200) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %202 = "stablehlo.subtract"(%201, %195) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %203 = "stablehlo.constant"() <{value = dense<-2.8276239805165836E-16> : tensor<f64>}> : () -> tensor<f64>
    %204 = "stablehlo.broadcast_in_dim"(%203) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %205 = "stablehlo.add"(%202, %204) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %206 = "stablehlo.multiply"(%174, %205) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %207 = "stablehlo.subtract"(%206, %200) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %208 = "stablehlo.constant"() <{value = dense<-3.425485619677219E-16> : tensor<f64>}> : () -> tensor<f64>
    %209 = "stablehlo.broadcast_in_dim"(%208) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %210 = "stablehlo.add"(%207, %209) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %211 = "stablehlo.multiply"(%174, %210) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %212 = "stablehlo.subtract"(%211, %205) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %213 = "stablehlo.constant"() <{value = dense<1.7725601330565263E-15> : tensor<f64>}> : () -> tensor<f64>
    %214 = "stablehlo.broadcast_in_dim"(%213) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %215 = "stablehlo.add"(%212, %214) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %216 = "stablehlo.multiply"(%174, %215) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %217 = "stablehlo.subtract"(%216, %210) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %218 = "stablehlo.constant"() <{value = dense<3.8116806693526224E-15> : tensor<f64>}> : () -> tensor<f64>
    %219 = "stablehlo.broadcast_in_dim"(%218) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %220 = "stablehlo.add"(%217, %219) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %221 = "stablehlo.multiply"(%174, %220) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %222 = "stablehlo.subtract"(%221, %215) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %223 = "stablehlo.constant"() <{value = dense<-9.5548466988283073E-15> : tensor<f64>}> : () -> tensor<f64>
    %224 = "stablehlo.broadcast_in_dim"(%223) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %225 = "stablehlo.add"(%222, %224) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %226 = "stablehlo.multiply"(%174, %225) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %227 = "stablehlo.subtract"(%226, %220) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %228 = "stablehlo.constant"() <{value = dense<-4.1505693472872222E-14> : tensor<f64>}> : () -> tensor<f64>
    %229 = "stablehlo.broadcast_in_dim"(%228) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %230 = "stablehlo.add"(%227, %229) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %231 = "stablehlo.multiply"(%174, %230) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %232 = "stablehlo.subtract"(%231, %225) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %233 = "stablehlo.constant"() <{value = dense<1.54008621752141E-14> : tensor<f64>}> : () -> tensor<f64>
    %234 = "stablehlo.broadcast_in_dim"(%233) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %235 = "stablehlo.add"(%232, %234) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %236 = "stablehlo.multiply"(%174, %235) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %237 = "stablehlo.subtract"(%236, %230) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %238 = "stablehlo.constant"() <{value = dense<3.8527783827421426E-13> : tensor<f64>}> : () -> tensor<f64>
    %239 = "stablehlo.broadcast_in_dim"(%238) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %240 = "stablehlo.add"(%237, %239) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %241 = "stablehlo.multiply"(%174, %240) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %242 = "stablehlo.subtract"(%241, %235) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %243 = "stablehlo.constant"() <{value = dense<7.180124451383666E-13> : tensor<f64>}> : () -> tensor<f64>
    %244 = "stablehlo.broadcast_in_dim"(%243) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %245 = "stablehlo.add"(%242, %244) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %246 = "stablehlo.multiply"(%174, %245) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %247 = "stablehlo.subtract"(%246, %240) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %248 = "stablehlo.constant"() <{value = dense<-1.7941785315068062E-12> : tensor<f64>}> : () -> tensor<f64>
    %249 = "stablehlo.broadcast_in_dim"(%248) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %250 = "stablehlo.add"(%247, %249) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %251 = "stablehlo.multiply"(%174, %250) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %252 = "stablehlo.subtract"(%251, %245) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %253 = "stablehlo.constant"() <{value = dense<-1.3215811840447713E-11> : tensor<f64>}> : () -> tensor<f64>
    %254 = "stablehlo.broadcast_in_dim"(%253) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %255 = "stablehlo.add"(%252, %254) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %256 = "stablehlo.multiply"(%174, %255) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %257 = "stablehlo.subtract"(%256, %250) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %258 = "stablehlo.constant"() <{value = dense<-3.1499165279632416E-11> : tensor<f64>}> : () -> tensor<f64>
    %259 = "stablehlo.broadcast_in_dim"(%258) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %260 = "stablehlo.add"(%257, %259) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %261 = "stablehlo.multiply"(%174, %260) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %262 = "stablehlo.subtract"(%261, %255) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %263 = "stablehlo.constant"() <{value = dense<1.1889147107846439E-11> : tensor<f64>}> : () -> tensor<f64>
    %264 = "stablehlo.broadcast_in_dim"(%263) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %265 = "stablehlo.add"(%262, %264) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %266 = "stablehlo.multiply"(%174, %265) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %267 = "stablehlo.subtract"(%266, %260) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %268 = "stablehlo.constant"() <{value = dense<4.9406023882249701E-10> : tensor<f64>}> : () -> tensor<f64>
    %269 = "stablehlo.broadcast_in_dim"(%268) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %270 = "stablehlo.add"(%267, %269) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %271 = "stablehlo.multiply"(%174, %270) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %272 = "stablehlo.subtract"(%271, %265) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %273 = "stablehlo.constant"() <{value = dense<3.3962320257083865E-9> : tensor<f64>}> : () -> tensor<f64>
    %274 = "stablehlo.broadcast_in_dim"(%273) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %275 = "stablehlo.add"(%272, %274) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %276 = "stablehlo.multiply"(%174, %275) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %277 = "stablehlo.subtract"(%276, %270) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %278 = "stablehlo.constant"() <{value = dense<2.266668990498178E-8> : tensor<f64>}> : () -> tensor<f64>
    %279 = "stablehlo.broadcast_in_dim"(%278) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %280 = "stablehlo.add"(%277, %279) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %281 = "stablehlo.multiply"(%174, %280) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %282 = "stablehlo.subtract"(%281, %275) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %283 = "stablehlo.constant"() <{value = dense<2.0489185894690638E-7> : tensor<f64>}> : () -> tensor<f64>
    %284 = "stablehlo.broadcast_in_dim"(%283) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %285 = "stablehlo.add"(%282, %284) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %286 = "stablehlo.multiply"(%174, %285) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %287 = "stablehlo.subtract"(%286, %280) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %288 = "stablehlo.constant"() <{value = dense<2.8913705208347567E-6> : tensor<f64>}> : () -> tensor<f64>
    %289 = "stablehlo.broadcast_in_dim"(%288) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %290 = "stablehlo.add"(%287, %289) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %291 = "stablehlo.multiply"(%174, %290) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %292 = "stablehlo.subtract"(%291, %285) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %293 = "stablehlo.constant"() <{value = dense<6.8897583469168245E-5> : tensor<f64>}> : () -> tensor<f64>
    %294 = "stablehlo.broadcast_in_dim"(%293) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %295 = "stablehlo.add"(%292, %294) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %296 = "stablehlo.multiply"(%174, %295) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %297 = "stablehlo.subtract"(%296, %290) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %298 = "stablehlo.constant"() <{value = dense<0.0033691164782556943> : tensor<f64>}> : () -> tensor<f64>
    %299 = "stablehlo.broadcast_in_dim"(%298) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %300 = "stablehlo.add"(%297, %299) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %301 = "stablehlo.multiply"(%174, %300) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %302 = "stablehlo.subtract"(%301, %295) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %303 = "stablehlo.constant"() <{value = dense<0.80449041101410879> : tensor<f64>}> : () -> tensor<f64>
    %304 = "stablehlo.broadcast_in_dim"(%303) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %305 = "stablehlo.add"(%302, %304) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %306 = "stablehlo.subtract"(%305, %295) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %307 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f64>}> : () -> tensor<f64>
    %308 = "stablehlo.broadcast_in_dim"(%307) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %309 = "stablehlo.multiply"(%308, %306) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %310 = "stablehlo.sqrt"(%4) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %311 = "stablehlo.divide"(%309, %310) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %312 = "stablehlo.constant"() <{value = dense<8.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %313 = "stablehlo.broadcast_in_dim"(%312) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %314 = "stablehlo.compare"(%4, %313) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %315 = "stablehlo.select"(%314, %172, %311) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    "stablehlo.custom_call"(%315, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    "func.return"(%315) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x7CC6A947E64A09C02087FB052D6AF03F5E0F9874DEFF11C0081FB974BE30E0BF031C6A91B3FE0440A56D526B7F5AD13F7BA60B82012E17C0CBD22C672CC8FE3FBD97FA0B6815F83F565EB6F5FBB2FABF5ED20C7484B90240C6CBC5066BD20840CC7AFF9538D104C07E951C153D3DE13FB29F3C15CE8F0F4009FFE051999D14C06551B1F4B53D0C40DF233D7CDAB5FC3F0F724601A1570AC0DE2D13959476FEBFA006098A07FB1D40C406E95ED5611740F2E45993CA8F16409804DEAC5A2CD33FB5E19DCE6C1CF4BF9F2479837DE41740121E6D36D828CFBF441A3B3A5A83CEBF70918288DEA8EB3FC78B0A43214CF93FB8FB84E394770AC03BC6A3B6D0300640B5A70ABEFB7A17C00BD2AD13497612C08F34C7E34C010140863A3FE6E969EC3F26A83576CC84FDBFB5E0AF770C7603C096020ADC1B5E01C0B47CEFD9762F094010EA304F23901540A6968C5919430A4076583C4988DF19C0770F406103C306408261E1AA34ACFD3F47899A1D06271540C4966BC8843403C098882BF53B9FE23F7DDA34CEDD560FC0922E9E1EC3AE1740D23C0B6353F500C0A88AA948B9190C4030C19DCE3EEA06C028E208EFD4A305C0D05D55D2EE0601C0645E9452D9B51DC0FF4C949D584215C0DA12F7F1F764F53F5645D834C22807C076EBA7C75F6C104019E18819DF9615C03EE72C7A6B80E63FB611C1FE097DFE3F94E8155978950A40B69C4F6D48C7B4BF72DFB393897B1140A1234F5998301BC0E009C7D3E932D03F604D7A9EB852E7BF5B4382A76EB001C0AE720ED046AE0AC067E5C4DF3985F93F98A0AAA3873004C04C97F41C07720740F727D2620A8401C03FA83393B9471540BC8EC732587D04402783A14C93B40C40C4F3FEB38E0FEBBF9C02642CF4EEE93FBC8C5D2AC5E7CCBF80CA489AC2480440F605DCB865590740B2927F90AD31FABFC7B36654FE1310400263F343C1A62140CA2BA05C9691F93F6886E318E2E108C0812C234DA635A3BF189B9EA41F49D13F828FE678669CF3BFBB75F9642C6205C09C23D5FCCECE0940B83B1B344C71034050AA46AFEDA3D8BF42EF74865A031DC03123EFAD2D8E0240DE41401E2E35164062F5E36C681B05C0C2F8EC8BAD3BD9BF0905C0076C0E1340A8AFCC53D74E14C00AC0F41744950DC01CA5F6B4986FDE3FFEB9961C4F15134018C21731A39EEFBF64EA3C2FC79C0040D99FB1F8FF28F7BF7E0C55574E46D73F08E74A866D7ADA3F034C2E5C12EE194034288D2E2205FA3FFBE9B6FAD1830640D607C1D72B4A15C0F51B0D823F221940F4EB519A2E3E02C04E08915BB6111140EAD6C4DA6BE4F5BF178D4A66C00EF3BF605B63975D61FB3FE8B4462A6BCE0A40C33E1BDF5BF11340D0278B05B42A18404A8150D9D66DC73F1AAA1E53E22322406DC705E09ADBF4BFCB6898C2A68EF3BF6ADD82ADD4800D409C437738624DD43F7B45EF027452E23F8A773C20F398FCBFB38EEB2DC329BE3F8C036102532313409EB8435674BCFA3FA727029AFF1EF3BF8BD13189ABD8E83F404E8BE20C6B02401A02C903A29B00C02EBB8A1AF598F1BF002F43FE9421EC3F039C229402EA973F7EFDFC6F11DAF53F9BF394A4FC95C53F4BC82B25D2290340BB94866C76E5F1BF051EBD941088C1BF9A79D06DF7A0F33FEC867D247FD6EB3F67E9CBC8A2471B4015E948A4B12FD23F2011744811E4E43FBEA47DA2113BE33F38EE3A1C699E15C086A2E9C1C620C9BFF60D93130CADF3BFEC8C0DF22E677FBF11B8D0265E4116C0E62F7A2A573110C07DBE1F5EF47112C0624FE85F71C710C047E29A779D43F7BFECBD82688D080B4026CA9C541B70FB3F3198F0731951D03FADDEDDBDAEE11040ECD4E7BCC9B7E5BF92C29CCE01950AC0CE15FE0515B8E83F276FCBA530B2EB3F3A2616B3B5C5F23FB9E6233FBC17014081035E26B6DCF53F700AD0FBD5D703C00BEC2EAEF254FD3F765A0028CD08F93FAFA9CF1E99C9F93FE2DD05CB594414C0387FABB631401040022DBD24EF8A10C0C301A97505E310C077A0D933CF85F5BFB61A79FEB22AFEBF6E3DA137CE1BCA3FE6FD3C6207D8DEBF263B18795CA00CC04ECB348CBE76E93F06481A4D880DFC3F1DA788269329E73F4D71F4AA344B02C0A8BC641DE4DB1240452907DA8BEF10C03D15EAD46A4406C0EA37DA6DF913F93FFE2AA3E40B71FFBFEB5248A62308E1BFA459DA3FBB2AE93FDCEF5DE07D61A53F494E8567239304C0E41B9BD2236A893F247BEF1D274E11C027C2B6245BF50040723E504D594C014040756D40A579EABF38B2CB51000107C09CF6CBEF5E1506C0A2A4C152DABFFD3FA28A181DFB3104C09AA9CE2E527401C057D4D70CF90C064024D556157180BD3F9689B6624CFCC6BF3808CFEF92EBF0BF4447BDF7866106C072E062CDF513D1BF62B592B0F0D500C05F47E55F0A5D134068B66B4CB51EF93FE27FA10B8656D83F48185E77948BE63F18D3A89959DE1F40C966ED1164FBF1BF1A589E70539D0740B2FC79C21D16A5BF697BD5FDB3E30240665DA6FF4FDBED3FDCFAB8D5FAA407C05A9B2B7DB4200FC0735B1C0195D709C03F48120E702314C0E6648D269E020D4004747C5D0B0AA83F4D264B2165EEF2BFF477EDF8E2271AC0F9F3C7114D37D83FA99A915CFBAD1040A8E59FC192EAFEBFF8678B7D2F5111401A8F109ED91B1840E7FAE4CD0D52F63F508324107C43E3BF78974DDB2058A83FA151B4875616E1BFDB6B573B468AF0BF950BC28796A8F6BF04D3F56C174BF0BF032DF2D0773AFB3F4C7251CD8A01FE3F8E6AB68816DA01407A0D4ECCAAC90F40A8CE24165834FB3F62A4BD12519900C06A7CD251DBF5503FA5AAF539498C06C01DDF4F6F1A6811C08F05A76FB1D4CCBFB64284759FAC07C0A54973FC0C6AF53F7A2F9BFDCA5C01404AA03658AB8FF4BF18533B834E8216C04022BE621C26F5BF88417D854C1F044003A415E60AB81440B3CDD2722CDF1240EA8F14011A74FABF6D08E355FADD0040FDC2F38C0C7AED3F289EC74A5FDF09C05513760E4269F13FEF1F62D0C88BEE3FFC170B29DFD5164098295144D82311C08B7CC0E9FC2BE9BF463A207B493E01C054E85A44E120D13F1C8FA001E1CC134096D23E7A3B8508C069BA268D2FEE14403CF653279C1FE1BF8D0802657A79F53FC88E450E31DBE03F17976EF9AD2F03C0574AD0CFDC0709C063F00F0E0E4713C092AB81F528CCE43FD8138457DBB50240BDCD716D2B8805409738CE0F8601F93F3E0DA130A4EF02C0C4E41355EB69FB3FB00E1D648C07F03F351AF8A95A81FA3FAA3327D60A7E02C06E9E12408D54F0BFC6A6D6E07E03E33F023336577E1703406383CB4EDD3308402743EC43E748C4BF41736EB617E6FFBF8AEFD62AF8C215C0DC943F90CB19C13F6D49FFB0903F04C02ED07FD75F4013C0ACB504CDD6F0EF3FCA1FAB4DC95C0BC0592BDB964A1E06C00226F661E57415C00BDDFA806D87FC3FBC5674912B5108C02266D53C4301C23FD1FEDCF858F504C00A5F86D498D60AC054EAC442C04FD1BFE84D48FC0C8A0E4009B8C2310DBBC33F837A17AABA761440EB12CB92293B0740DAFC5F8229F500C06B1F7B493E131AC071F0FD2B9AA614C09FF28826ED69F93FE7A02E7D732F15C0B0CB7A88C054EFBF38049E6870CEC63FC5B49740F0C512C0FC7B1DC05644C9BF04081EB0A48F0440C2DB50BEF384124039204DED29A30DC0D3BD47FF3AEC184076137DDE1485F5BFF2AA5C2155941E40C4F09FC1E39601C0AEB31ECE3683064056EE1FD50FAB13408D29E82DBC4CE83F283BDB40842FF3BFE99B5415561204C0DE051582A37BF93FC1362A3FE5D8D43F762E48EA1EC71140266C79889E0F06C05C13C56435CA04C02E34F27CCF45FC3F685A78E0FD440C40F2034035D05D1AC0A63560218CE60C4084A164D68130CA3F70F6944530D41540D162D12F6413D1BFC835FFC2C289EF3F48B74D3B00C8FEBF0623094210D704403F5792D469C40FC05E56D4131D6E104070939818815B05C05098D6B4247FEABF86758305586AFA3F144D76F7077709409A2BF053D0EA04C0136B936131A90E402AC6B9929A71FB3F8FCEF6E0E621F0BF0A199E7CC7F10DC00DEF132BBA050440245E9528F058BABF813003534939D13F1DF6C209A55B02401C9E3BC3E75F1340F66C8A11D2D2FFBF448F51C40BE4F53F10F460A7724D1140FE33579D2335174038BF979FC21FFC3F826A7E50BAF8ECBF406C7ACE49D0F73F1EBE393F074C07C0EEA4DB9ED9FCE23FC5691834F8552040A64F7BD1F1930240A170ABC83C39E63FEC7AC874EBF40AC0A5094DF526BDF63FE251972B56BA05C0CE2DD149DAC50840E2CDDFD0A54D1D40CFBA2FD9A6FB00409C9F4A090B73E8BF48AB1966FB1E1C401407A1265F101AC02597E5BF7DD3B53FEBC4C4BDEDF9D43F956DE906959D16C06FD62E45EB470CC0863787A3A3720FC0FE1201BBB3BF0C40AB282D157F780440945D435D3CDC18C05B7DFE0A718CF23FE1966B3D2D9E0C40"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%1) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x8CCE3B230F31CE3F0639172D2B63DD3FD958F52B96DBC83F8117E870688CE43FF23E853C80CCD03F0B68BF4AC4D9E83F9A449D46EEB9C53F829CC5D9C236D43FADE8CA0A5C77D73F58D0DE867F0AD63FBA881A5DF9F7D13F9A5604DAA983CE3FA1543058DFE1D03F378B1BD01E0DE43F458D1E7057B2CA3FD7B85BDDF81DC73FDAB840EF1662CC3FFCEEFA3C7716D53F43306EAF0982CD3F59F026968757D43FE367B5F0A5FDC23FCED1893781A0C53F5DECE7A3CA09C63F273814424F40E83F15FD424AD824DA3FA8930C3BEF61C53F51BCAA8C7575E93FECD485D38092E93F720D7B94D91EE03F131159BAB7C6D63FFC82FB54066ECD3F3AA3FF477844D03FA461A4E54A94C53F223BCB505685C83FC0203F085F08D33F809AE9F0F8C9DF3FEA272C3114BCD43FC20FC2412A90D13F96C0ABB115CBD23F3EE9B164A243CE3F1F5470C48392C63F23FB1495FB8ECD3FC954BE8FE781C43F32791A7DEC07D03F8DBD8C2854ABD43FF2DC4AF8B7CDC63F3AB4EF2176B3D13F996D60CAA86DE33F8A1E762ED6CCCA3F9090B388677BC53F1E12E6317110D33FF92197884376CC3FE02370EB3FF0CF3FF7E7A2367D81D03F8CB8E1869604D33FA855602E9414C33F6E57332927BEC63F0ED95B224F33D93F4262A69EA7BECF3F7231CC302E1FCA3FFBC6F8AEC88EC63F6EE6B15A1ADCE13F4A44C838EA54D43F38F9E169735BCD3FAEAB88AECD8DED3F6C730C3E6340C93F4BEE5F276FFBC33F12E484AA693EE93F0753654DED8EE13FB6A402C08896D23F04517D27244CCD3F8E92BF26AFA7D63FFA83F32B0130D13FD054246DAC85CF3FB498B837AAB2D23F5C744E7D1ABBC63F96F76448220AD13FB6179B15A720CC3FE0E7F729064EE03FB8862714E0A9E03F24989D49A8DBE93FF8D1C200F423D13FB8F6045FB198CF3F38A7B87D7B4CD63F601F28A1096DCA3F177747AF3473C13F6077527405A1D63F4A51BABCE878CE3FCA11F53F1ED5EE3F4F9030719CDFE83FB704EAFFDE88DA3F0B68216CD79ED03F20CE2D199DD9CD3F1E251F67B292D13F8D5AB069B596E63FC09EA6174851C33F20082CE0D810D23F51830D4B1F39C63FB2A7F5672CBFD03F10FCC44F8D6BE63FDAA764D3951BC83F97D92CC6804DC73FE604D52CBEA9CB3F05CE0BD29807E53F468446EDEC16C83F9BDD01C99201DE3FB82C25D6534DD33F5D51431A2307D83FE52028E149FCE63F3F1D2D13B112E63F87EC63E7E17BC43F79DE90CFAE63D63FDDD881FEC321D03F4E7ECCD5B7B9C63F8E612F8C42D2C43F445343E6CE3FD23FA819E13F9B94C93FEC5C107519DBD83FA410D77DADFBDA3FBAE480469BB4D53F58A878267238CD3FAEA36F6F7487C73FEB49CB652E41C53FDEE8866906DFEA3FD9FFAD7FB234C13F280B67FEBC95D93F3289702FD293DA3F21D0AB294FB4CB3F7FA734882BE4E73FEE536B6F778FE33F2E161B6B6C23D53FA0697880C48AEC3FA1E732CE790DC83FD81D0F63BE05D63FEC6D314948EEDA3FB2CC066E6206E13F80238C744C25D23F8148140C214ED33F325A97FD2541DC3F74116795F0F4DF3F10219CBAFE43EF3FF4AD70F527E2D83F01A9A79F003AEB3FC69E142E4DB9D13F58578573BCFBDB3FAE8181D0F808EC3F3CA681543E85DA3F7A6867520511E03F4F3FBEFD9DF2C33F140D2948C492E83FE6E9A4D5187BE23FE4D4B94C542AE33FDC51838E9D8AC63FD192A670D58CEA3F6E40BBF2AA7BDA3F7A654AB18DC1EF3F17FE4D9BAF32C63F93201CB8E052CA3F82A418D56D88C83F18C797D5C1D1C93FCDA6D07485F6D73F020CFC193915CD3FF17637D180ADD53F0007EFD20334E93FBC0AB309F4BBC93FE91DDEFF2828E23F702048CEBC5BCD3F72A0CAF97A11E13FBE824B63041CE03F6CBE6D3DA538DB3F384E2E145AF9D23F234A75415AE0D83F0CD507F8F85CD13F200C0BFF9CD0D43FA2AC74B3DDEBD63F58EF7B9A1383D63F3422F605EC53C73F8FD2343DC145CA3FE4B82309E404CA3F0C7CB9C3D8BAC93FC3473C724C1CD93F1373D1AA8576D43FBAE3E3501C5EEA3F22D7515C55EDE43F592DA541A72BCC3FE4E8DF6759D1E03F45FA41811E63D53F5DF2D38ED39DE13FBA3D846E1138D23F64DDA90A1B3EC83FC493F08D85B0C93FC4BA2DDC313CD03F8E268ED0A8E5D63F2A9516BFB6F4D33FB6490729D925E43F635B86BCB0EAE03FB823CE7B60B4EE3F9C16CDA890FFD03FFFAC29FE479BEF3F9FC05F421964C93F524765F46B10D33F206DC352A4D6D23F6612A79C3A7DE03F80A4EA1117DECF3F9E0DACC52350D03FF691E70408A3D43FD6267BB3472FD13FAB741FDABDBCD23F88E71902BB53D03FC8D9D2B1919CEC3F56B16C39BEF4EA3F87C804A524E4DC3F7BBD2342FF2FD03F3C2E68A48CF1E83F82DD8135C725D33F4C3B173605E7C73F1960D069B6DFD63F052BDF0BF0ACE63F3F7DC85BF1D7E13F31CE91EA0166C23FD487C3721DE8DB3F7A08EBFE9064CF3F6986C2FFCDB8EE3F767B259622E0D13F455351F2A8F5DE3FDD9721C1C15ECF3F0C11D77C55E6CA3FEC7FEE83E7D3CD3FDA216EBA3268C73F9BEAEBE0A8F6CB3F0CB7CBFB918CEE3F5F69C8C08816DB3F55FAD3FA2364C43FD6242C1AF2B5E63F9B83DB5F20E7C93F1E91AC8E1D29D43F6445938CB161C93F8492D5820F48C53F4176E1038691D83F8A7CD9CEBD26E33F5A5050CB0588EE3FF5269100371FE43F0B73D93A3743DD3F90B3A2BAE558D83F53F73E646582DD3FDE866F5E77C7D53FAC90378E8B87D43F2A2ADFF4907CD23F97FFC744BC97CA3F3A87277273CAD53FC45BC960C04FD33F50E392C186F7EF3F9DCE0268461ED03F0EE8FD1D9B4FC93F4D8B758D14DFE93FCDD261E0F758CF3FBF5E6E38BC2FD93FCCC40736F0CBD23FF94745D9B4CDD93FBA983F80C210C63F5CBCCA83E85FD93F9A868488A238D13F45E755ED440EC73FACA3AB62D83BC83FAE10C424562AD63F6B7605F24920D33F086F425F6E2CDF3F4C69D78AD8CECD3FAC67F8FB306DDC3FAA7773FA4794DE3FFA7457E9FDE5C53F26F49312F085C93F645B37E144EAE03F56EA0560D9DFD23FD571CAAF2EEDE83F5BD433D5899EC73FF41A7EFF05BACE3F109CE82A82EEC63F06718439E41AE43F6EC88661EA24D93F4A63DA21F33AE43FE4D622B519B6D13F88CC2B77B05ECE3FBCF6652995F5C73FE3B75AC6A384E23F7AE473520FFAD13F56EDE2EBC78DD03FCACC5FF8EAEFD63F9A48F34D74D9D13FAF3CDA597BB0D53F7A906F833AC7DD3FE458B1639623D63FE2CC5579351AD23F71F502CDDC78DD3FA4CFE9202542E33FAC1034B55BC3D13F6ABA301699F4CE3F926DF7C3617BEB3F573427E049C8D33FC3EE9CBC8776C63FD3740170871FEC3FB3CEDCCD8328D13FDA9ED54A07FAC73FE8464B20D5D6DD3FE160A6AD13E3CC3F6F3FC5F2554CD03F952215EEAEA1C63F8A02D6CF512BD53F707AB13A59DFCE3F3A6A3AA44EF0EB3F3E8B61DADED0D03F12F3A9B37533CD3F38B2BE7661DDE83FB0DA4C89D02ECB3F1F08815D8597EB3FE0A65C354935C73FAC4BDD2E3AB0CF3FDEBFF97C8D10D33FDE094C9C946CC43FEE4A638B9C18C73FDC8A842178B6D63F92E2A274E7C8C63F1AA97C576E28DE3F46234CC98BFDEA3FB38AB07A494DC83F9825F2DE2E86EA3F03D59F3B4101D13FAD1545A0E67AC83FC213DF9695A2CB3F319164A8E1E9C43F16BABC74CE1CD93F3A0AB7CCF7CBC23FF5729077A9A6D23F3EF506100422D03FDCADE1D72DB4C73F714AA34F6E36E13F334DCC77B9E0DA3F9130DE21293FD13FF7301092DDACD63FBFA9E8FC82B8E73FD2D2B2E03806C93FBED094139952D03FB9D6E12D32E5D03F94FF4B9F2A49D53F2EC6425A075ECC3F37B1B606494EC43F7CDA3E56AD05CC3F04AA91D2475AEA3F7A879CAA246DC63FE7CD00E4BDF1E83F5C45AC9E820CDE3F7EFCD76FD436D43FC2AFB8F51BDFD03F12EFCA5B239ACA3F1A0CAEEEAC1DCA3F5D17417ADBA1D03F9A36BA857A7BE03F7092D4B6512FD63F115D42997613CE3F4D9585A7CED5D03F914847DCA31FCB3F9679DD7CC8ACD53F3CB24E0E36ACDD3F0951722A817ACB3FD6A3053B8945D13FD235B33267F2EC3F1D993437F2E4E83FD5E331F1592ED23FD2B7BF6D21E5C73F33DAF17B87CFD33FA43374DD5ADBD83F26CE1D7AA864C93F624BFE0969B6C53FAA72C25EAD5AD53FFAB4F2F77E76DF3FE0896B7285A0D73F1C3474D012A3CF3F4FE177E20145E33F86CDA116AF29C23F054C37E2830DD23F18DD829CCFF6E13F1A772DB70F21CD3F55C689B99F4BD83F8FC325599177D03FB17AC04F6F8CCE3FFE2C7BA2BD37C33FF8BDC5742C0CD33F1DCBE57F2C29E13F11829AD15DA2C33F81C15501C26DC43F341602772F70ED3F88D23B4440AEE73FBC009E25B102C63F020880E3655CCC3F4CA893BADFBFCA3F482BBF909E1ACC3F48D974D47E0CD13FA9F167CDEFF0C43F60C8EAB65069DB3F003FBE62D82CCC3F"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%0) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

